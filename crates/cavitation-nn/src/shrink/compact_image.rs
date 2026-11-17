//! Compaction of image models: [`compact`](super::compact) on a model with `Conv2d` layers
//! removes the output channels its channel masks switch off, from the conv, the layers of
//! its run and the layer that receives them, and the neurons the neuron masks of its dense
//! layers switch off, as it does for a dense network.

use cavitation_core::kernels::pooling::Pool2dGeom;
use cavitation_core::{Conv2dGeom, Error, ParamStore, Real, Rng};

use super::compact::{
    check_narrowed_run, kept_neurons, placed_masks, rebuild_linear, rebuild_prelu, run_owners,
};
use crate::inference::InferenceLayer;
use std::sync::Arc;

use crate::{
    AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv2d, Init, Layer, MaxPool2d, Sequential,
};

/// What one conv loses: the conv's layer index, the last layer of its run, the values per
/// channel there, and the channels kept, in order.
struct Narrowed {
    /// The `Conv2d`'s layer index.
    conv: usize,
    /// [`Sequential::channel_layer`] of the conv.
    end: usize,
    /// Values per channel at `end`.
    plane: usize,
    /// The output channels that stay on.
    kept: Vec<usize>,
}

/// [`compact`](super::compact) for a model with image layers: see its documentation.
pub(super) fn compact_image<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
) -> Result<(Sequential<T>, ParamStore<T>), Error> {
    let owners = run_owners(model);
    check_layers(model, &owners)?;
    let narrowed = narrowed_convs(model, params)?;
    // The dense layers' neuron masks, checked and resolved as for a dense network.
    let placed = placed_masks(model, &owners, true)?;
    for (&linear, masked) in &placed {
        check_narrowed_run(model, params, linear, masked)?;
    }
    let keep = kept_neurons(model, params, &placed)?;
    // The neurons kept of the `Linear` whose run holds layer `i`, if it loses any.
    let kept_of = |i: usize| owners[i].and_then(|owner| keep[owner].as_deref());
    let mut store = ParamStore::<T>::new_generic();
    // The copy fuses (or not) as the model does.
    let mut out = Sequential::<T>::new_generic().fuse(model.fuses());
    // For each layer, the `Narrowed` whose run it is in, if any.
    let in_run = |i: usize| narrowed.iter().find(|n| (n.conv + 1..=n.end).contains(&i));
    // The `Narrowed` whose run ends right before layer `i`: its kept channels are the
    // inputs layer `i` keeps.
    let feeding = |i: usize| narrowed.iter().find(|n| n.end + 1 == i);
    let no_params = ParamStore::new().values;
    for (i, rows) in keep.iter().enumerate() {
        let layer = model.layer(i);
        let own = narrowed.iter().find(|n| n.conv == i);
        if let Some(conv) = layer.as_conv2d() {
            let ins = feeding(i).map(|n| n.kept.as_slice());
            let outs = own.map(|n| n.kept.as_slice());
            out = out.push(rebuild_conv(i, conv, params, &mut store, outs, ins)?);
        } else if let Some(bn) = layer.as_batch_norm2d() {
            let kept = in_run(i).map(|n| n.kept.as_slice());
            out = out.push(rebuild_batch_norm(bn, params, &mut store, kept)?);
        } else if let Some(linear) = layer.as_linear() {
            // The columns kept: a conv's kept channels as blocks of inputs after a flatten,
            // or the on neurons of the `Linear` whose run ends just before.
            let cols: Option<Vec<usize>> = match feeding(i) {
                Some(n) => Some(
                    n.kept
                        .iter()
                        .flat_map(|&c| c * n.plane..(c + 1) * n.plane)
                        .collect(),
                ),
                None => i.checked_sub(1).and_then(kept_of).map(<[usize]>::to_vec),
            };
            out = out.push(rebuild_linear(
                i,
                linear,
                params,
                &mut store,
                rows.as_deref(),
                cols.as_deref(),
            )?);
        } else if let Some(prelu) = layer.as_prelu() {
            out = out.push(rebuild_prelu(prelu, params, &mut store, kept_of(i)));
        } else if let Some(n) = in_run(i)
            && let Some(pool) = layer.inference(&no_params).and_then(|l| narrow_pool(&l, n))
        {
            out = out.push_shared(Arc::from(pool?));
        } else {
            out = out.push_shared(model.shared_layer(i));
        }
    }
    Ok((out, store))
}

/// Refuses, naming it, a layer with parameters that compaction cannot rebuild, and a mask
/// outside every `Linear`'s run (`owners`) that is not a whole-channel mask at the end of a
/// conv's run. The masks in a `Linear`'s run are checked as for a dense network.
fn check_layers<T: Real>(model: &Sequential<T>, owners: &[Option<usize>]) -> Result<(), Error> {
    if let Some(i) = foreign_image_parameters(model) {
        return Err(Error::Unsupported(format!(
            "compact: layer {i} ({}) has parameters, and only Conv2d, BatchNorm2d, Linear and \
             Prelu layers can be rebuilt in the new store",
            model.layer(i).name()
        )));
    }
    for (i, owner) in owners.iter().enumerate() {
        let layer = model.layer(i);
        let Some(mask) = model.mask(i) else { continue };
        if mask.off().next().is_none() || owner.is_some() {
            continue;
        }
        let conv = (0..=i)
            .rev()
            .find(|&c| model.layer(c).as_conv2d().is_some() && model.channel_layer(c) == i);
        let whole = conv.is_some_and(|c| {
            model
                .channel_mask(c)
                .is_some_and(|m| model.channel_values(c, &m) == *mask)
        });
        if !whole {
            return Err(Error::Unsupported(format!(
                "compact: layer {i} ({}) carries a mask that is not a whole-channel mask at the \
                 end of a Conv2d's run; in a model with image layers compact removes the \
                 channels switched off with set_channel_mask and the neurons of its Linear \
                 layers; clear the mask with clear_mask({i}) first",
                layer.name()
            )));
        }
    }
    Ok(())
}

/// The first layer of `model` with parameters that is not a `Conv2d`, a `BatchNorm2d`, a
/// `Linear` or a `Prelu`: compaction cannot move its parameters to a new store. One of the
/// layout checks of image compaction, shared with the shrink tool.
pub(super) fn foreign_image_parameters<T: Real>(model: &Sequential<T>) -> Option<usize> {
    (0..model.len()).find(|&i| {
        let layer = model.layer(i);
        let rebuilt = layer.as_conv2d().is_some()
            || layer.as_batch_norm2d().is_some()
            || layer.as_linear().is_some()
            || layer.as_prelu().is_some();
        !rebuilt && !layer.parameters().is_empty()
    })
}

/// The first layer in the run of the `Conv2d` at layer `conv` that compaction cannot
/// narrow: one that is not a `BatchNorm2d`, a pooling, an activation, a `Prelu` with one
/// slope, `Flatten`, `Identity` or `Dropout`. One of the layout checks of image
/// compaction, shared with the shrink tool.
pub(super) fn unvouched_channel_layer<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
    conv: usize,
) -> Option<usize> {
    (conv + 1..=model.channel_layer(conv)).find(|&l| {
        let layer = model.layer(l);
        let pool = super::POOL_NAMES.contains(&layer.name());
        let passes = matches!(layer.name(), "flatten" | "identity" | "dropout");
        // A Prelu with one slope treats every channel alike; one with a slope per value
        // would need its slopes narrowed too.
        let one_slope = layer
            .as_prelu()
            .is_some_and(|p| params.value(p.weight()).data().len() == 1);
        !(layer.as_batch_norm2d().is_some()
            || pool
            || passes
            || layer.activation().is_some()
            || one_slope)
    })
}

/// The convs that lose channels, after checking that each one's run can follow and that a
/// `Conv2d`, or a `Linear` after a flatten, receives the channels.
fn narrowed_convs<T: Real>(
    model: &Sequential<T>,
    params: &ParamStore<T>,
) -> Result<Vec<Narrowed>, Error> {
    let mut out = Vec::new();
    for conv in (0..model.len()).filter(|&c| model.layer(c).as_conv2d().is_some()) {
        let Some(mask) = model.channel_mask(conv) else {
            continue;
        };
        if mask.off().next().is_none() {
            continue;
        }
        let end = model.channel_layer(conv);
        if let Some(l) = unvouched_channel_layer(model, params, conv) {
            return Err(Error::Unsupported(format!(
                "compact: layer {l} ({}) is in the run of the conv layer {conv}, and compact \
                 cannot tell whether it treats every channel alike once channels are \
                 removed; use batch norm, a pooling, an activation, Flatten, Dropout or \
                 Identity there, or clear the channel mask of layer {conv}",
                model.layer(l).name()
            )));
        }
        let next = end + 1;
        let receives = (next < model.len())
            && (model.layer(next).as_conv2d().is_some() || model.layer(next).as_linear().is_some());
        if !receives {
            let what = if next < model.len() {
                format!("layer {next} ({})", model.layer(next).name())
            } else {
                "the model's output".to_string()
            };
            return Err(Error::Unsupported(format!(
                "compact: the channels of the conv layer {conv} reach {what}, which is not a \
                 Conv2d or a Linear, so they cannot be removed; clear its channel mask first"
            )));
        }
        let plane = model.channel_shape(conv).1;
        out.push(Narrowed {
            conv,
            end,
            plane,
            kept: mask.on().collect(),
        });
    }
    Ok(out)
}

/// `conv` rebuilt in `store` with the output channels `outs` and the input channels `ins`
/// (all when `None`), its weight and bias copied.
fn rebuild_conv<T: Real>(
    index: usize,
    conv: &Conv2d,
    params: &ParamStore<T>,
    store: &mut ParamStore<T>,
    outs: Option<&[usize]>,
    ins: Option<&[usize]>,
) -> Result<Conv2d, Error> {
    let g = conv.geom();
    let all = |n: usize| (0..n).collect::<Vec<usize>>();
    let outs = outs.map_or_else(|| all(g.out_channels()), <[usize]>::to_vec);
    let ins = ins.map_or_else(|| all(g.in_channels()), <[usize]>::to_vec);
    let geom = Conv2dGeom::with_options(
        ins.len(),
        outs.len(),
        g.input(),
        g.kernel(),
        g.stride(),
        g.padding(),
        g.dilation(),
    )?;
    let w = params.value(conv.weight()).data();
    let (kh, kw) = g.kernel();
    let taps = kh * kw;
    if w.len() != g.weight_len() {
        return Err(Error::InvalidData(format!(
            "compact: the conv layer {index} expects {} weight values, the store holds {}",
            g.weight_len(),
            w.len()
        )));
    }
    let mut weight = Vec::with_capacity(outs.len() * ins.len() * taps);
    for &o in &outs {
        for &c in &ins {
            weight.extend_from_slice(&w[(o * g.in_channels() + c) * taps..][..taps]);
        }
    }
    let bias = conv.bias().map(|id| {
        let b = params.value(id).data();
        outs.iter().map(|&o| b[o]).collect::<Vec<T>>()
    });
    let rebuilt = Conv2d::with_init(
        store,
        geom,
        Init::Zeros,
        bias.as_ref().map(|_| Init::Zeros),
        &mut Rng::seed(0),
    );
    store
        .value_mut(rebuilt.weight())
        .data_mut()
        .copy_from_slice(&weight);
    if let (Some(id), Some(b)) = (rebuilt.bias(), bias) {
        store.value_mut(id).data_mut().copy_from_slice(&b);
    }
    Ok(rebuilt)
}

/// `bn` rebuilt in `store` with the channels `kept` (all when `None`): γ, β, the running
/// mean and variance and the batch count copied.
fn rebuild_batch_norm<T: Real>(
    bn: &BatchNorm2d,
    params: &ParamStore<T>,
    store: &mut ParamStore<T>,
    kept: Option<&[usize]>,
) -> Result<BatchNorm2d, Error> {
    let channels = bn.channels();
    let all: Vec<usize> = (0..channels).collect();
    let kept = kept.unwrap_or(&all);
    let running = bn.running();
    let rebuilt = BatchNorm2d::with_options(
        store,
        kept.len(),
        (1, bn.spatial()),
        bn.eps(),
        running.and_then(|r| r.momentum),
        bn.weight().is_some(),
        running.is_some(),
    )?;
    let copy = |store: &mut ParamStore<T>, from: cavitation_core::ParamId, to| {
        let v = params.value(from).data();
        let picked: Vec<T> = kept.iter().map(|&c| v[c]).collect();
        store.value_mut(to).data_mut().copy_from_slice(&picked);
    };
    if let (Some(from), Some(to)) = (bn.weight(), rebuilt.weight()) {
        copy(store, from, to);
    }
    if let (Some(from), Some(to)) = (bn.bias(), rebuilt.bias()) {
        copy(store, from, to);
    }
    if let (Some(from), Some(to)) = (running, rebuilt.running()) {
        copy(store, from.mean, to.mean);
        copy(store, from.var, to.var);
        if let (Some(a), Some(b)) = (from.count, to.count) {
            let count = params.value(a).data().to_vec();
            store.value_mut(b).data_mut().copy_from_slice(&count);
        }
    }
    Ok(rebuilt)
}

/// The pooling `layer` (as its inference form describes it) rebuilt for the channels `n`
/// keeps, or `None` if it is not a pooling.
fn narrow_pool<T: Real>(
    layer: &InferenceLayer,
    n: &Narrowed,
) -> Option<Result<Box<dyn Layer<T>>, Error>> {
    let channels = n.kept.len();
    let geom = |g: &Pool2dGeom| {
        Pool2dGeom::with_options(
            channels,
            g.input(),
            g.kernel(),
            g.stride(),
            g.padding(),
            g.dilation(),
            g.ceil_mode(),
        )
    };
    Some(match layer {
        InferenceLayer::MaxPool2d(g) => {
            geom(g).map(|g| Box::new(MaxPool2d::new(g)) as Box<dyn Layer<T>>)
        }
        InferenceLayer::AvgPool2d {
            geom: g,
            count_include_pad,
            divisor_override,
        } => geom(g)
            .and_then(|g| AvgPool2d::with_options(g, *count_include_pad, *divisor_override))
            .map(|p| Box::new(p) as Box<dyn Layer<T>>),
        InferenceLayer::AdaptiveAvgPool2d { input, output, .. } => {
            AdaptiveAvgPool2d::new(channels, *input, *output)
                .map(|p| Box::new(p) as Box<dyn Layer<T>>)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    //! Tests for compacting image models: the compacted model gives the masked one's outputs
    //! within the rounding of the shorter sums, on each shape the run can take, and the
    //! refusals.

    use super::super::compact;
    use crate::{
        AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv2d, Flatten, InferenceModel, Linear,
        MaxPool2d, NeuronMask, Relu, Sequential,
    };
    use cavitation_core::kernels::pooling::Pool2dGeom;
    use cavitation_core::{Conv2dGeom, Error, ParamStore, Rng, Tensor};

    /// Random values for every parameter of `params`, and non-trivial running statistics.
    fn scramble(params: &mut ParamStore, seed: u64) {
        let mut rng = Rng::seed(seed);
        for (_, value, _) in params.iter_mut() {
            for v in value.data_mut() {
                *v = rng.uniform_f32() * 1.5 + 0.25;
            }
        }
    }

    /// The masked model's and the compacted model's outputs on `rows` random inputs agree
    /// within 1e-4 relative to the outputs' size.
    fn agree(net: &Sequential, params: &ParamStore, inputs: usize, rows: usize) {
        let (small, small_params) = compact(net, params).unwrap();
        let mut rng = Rng::seed(77);
        let x: Vec<f32> = (0..rows * inputs)
            .map(|_| rng.uniform_f32() * 2.0 - 1.0)
            .collect();
        let x = Tensor::from_vec(x, &[rows, inputs]).unwrap();
        let mut a = InferenceModel::compile(net, params).unwrap();
        let mut b = InferenceModel::compile(&small, &small_params).unwrap();
        let want = a.infer_batch(x.view()).data().to_vec();
        let got = b.infer_batch(x.view()).data().to_vec();
        assert_eq!(want.len(), got.len());
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * scale, "output {i}: {g} against {w}");
        }
        assert!((0..small.len()).all(|i| small.mask(i).is_none()));
    }

    /// conv → BatchNorm2d → ReLU → conv → ReLU → flatten → linear, with channels off at the
    /// first conv, the second, and both.
    #[test]
    fn conv_bn_relu_conv() {
        for offs in [
            (vec![1, 4], vec![]),
            (vec![], vec![0, 2]),
            (vec![0, 3], vec![1]),
        ] {
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(1);
            let mut net = Sequential::new()
                .push(Conv2d::new(
                    &mut params,
                    Conv2dGeom::with_options(2, 6, (6, 6), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(),
                    &mut rng,
                ))
                .push(BatchNorm2d::new(&mut params, 6, (6, 6)))
                .push(Relu)
                .push(Conv2d::new(
                    &mut params,
                    Conv2dGeom::new(6, 4, (6, 6), (3, 3)).unwrap(),
                    &mut rng,
                ))
                .push(Relu)
                .push(Flatten)
                .push(Linear::new(&mut params, 64, 3, &mut rng));
            scramble(&mut params, 5);
            if !offs.0.is_empty() {
                net.set_channel_mask(0, NeuronMask::from_off(6, &offs.0));
            }
            if !offs.1.is_empty() {
                net.set_channel_mask(3, NeuronMask::from_off(4, &offs.1));
            }
            agree(&net, &params, 72, 5);
            let (small, _) = compact(&net, &params).unwrap();
            let c0 = small.layer(0).as_conv2d().unwrap().geom();
            let c3 = small.layer(3).as_conv2d().unwrap().geom();
            assert_eq!(c0.out_channels(), 6 - offs.0.len());
            assert_eq!(
                small.layer(1).as_batch_norm2d().unwrap().channels(),
                6 - offs.0.len()
            );
            assert_eq!(
                (c3.in_channels(), c3.out_channels()),
                (6 - offs.0.len(), 4 - offs.1.len())
            );
            assert_eq!(small.linear(6).unwrap().inputs(), 16 * (4 - offs.1.len()));
        }
    }

    /// conv → ReLU → max pool → conv, and conv → BatchNorm2d → ReLU → avg pool → flatten →
    /// linear, and conv → ReLU → adaptive avg pool → flatten → linear.
    #[test]
    fn runs_through_poolings() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(2);
        let mut net = Sequential::new()
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(1, 4, (6, 6), (3, 3)).unwrap(),
                &mut rng,
            ))
            .push(Relu)
            .push(MaxPool2d::new(Pool2dGeom::new(4, (4, 4), (2, 2)).unwrap()))
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(4, 2, (2, 2), (1, 1)).unwrap(),
                &mut rng,
            ));
        scramble(&mut params, 6);
        net.set_channel_mask(0, NeuronMask::from_off(4, &[2]));
        agree(&net, &params, 36, 4);

        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(1, 4, (6, 6), (3, 3)).unwrap(),
                &mut rng,
            ))
            .push(BatchNorm2d::new(&mut params, 4, (4, 4)))
            .push(Relu)
            .push(AvgPool2d::new(Pool2dGeom::new(4, (4, 4), (2, 2)).unwrap()))
            .push(Flatten)
            .push(Linear::new(&mut params, 16, 3, &mut rng));
        scramble(&mut params, 7);
        net.set_channel_mask(0, NeuronMask::from_off(4, &[0, 3]));
        agree(&net, &params, 36, 4);

        let mut params = ParamStore::new();
        let mut net = Sequential::new()
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(1, 3, (5, 5), (3, 3)).unwrap(),
                &mut rng,
            ))
            .push(Relu)
            .push(AdaptiveAvgPool2d::new(3, (3, 3), (1, 1)).unwrap())
            .push(Flatten)
            .push(Linear::new(&mut params, 3, 2, &mut rng));
        scramble(&mut params, 8);
        net.set_channel_mask(0, NeuronMask::from_off(3, &[1]));
        agree(&net, &params, 25, 4);
    }

    /// conv → ReLU → flatten → linear → Prelu (a slope per neuron) → linear, with neurons
    /// of the hidden linear off, alone and with channels of the conv off: the dense head
    /// loses its neurons as in a dense network, and the Prelu its slopes.
    #[test]
    fn dense_head_neurons() {
        use crate::Prelu;
        for channels_off in [vec![], vec![1]] {
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(5);
            let mut net = Sequential::new()
                .push(Conv2d::new(
                    &mut params,
                    Conv2dGeom::new(1, 3, (5, 5), (3, 3)).unwrap(),
                    &mut rng,
                ))
                .push(Relu)
                .push(Flatten)
                .push(Linear::new(&mut params, 27, 6, &mut rng))
                .push(Prelu::new(&mut params, 6))
                .push(Linear::new(&mut params, 6, 2, &mut rng));
            scramble(&mut params, 9);
            net.set_neuron_mask(3, NeuronMask::from_off(6, &[0, 4]));
            if !channels_off.is_empty() {
                net.set_channel_mask(0, NeuronMask::from_off(3, &channels_off));
            }
            agree(&net, &params, 25, 4);
            let (small, small_params) = compact(&net, &params).unwrap();
            let hidden = small.linear(3).unwrap();
            assert_eq!(
                (hidden.inputs(), hidden.outputs()),
                (9 * (3 - channels_off.len()), 4)
            );
            assert_eq!(small.linear(5).unwrap().inputs(), 4);
            let slopes = small.layer(4).as_prelu().unwrap().weight();
            assert_eq!(small_params.value(slopes).data().len(), 4);
        }
    }

    /// A `Prelu` with a slope per value in a narrowed conv's run is refused, naming it;
    /// one with a single slope is kept.
    #[test]
    fn prelu_in_the_run() {
        use crate::Prelu;
        for (slopes, ok) in [(64, false), (1, true)] {
            let mut params = ParamStore::new();
            let mut rng = Rng::seed(4);
            let mut net = Sequential::new()
                .push(Conv2d::new(
                    &mut params,
                    Conv2dGeom::new(1, 4, (6, 6), (3, 3)).unwrap(),
                    &mut rng,
                ))
                .push(Prelu::new(&mut params, slopes))
                .push(Flatten)
                .push(Linear::new(&mut params, 64, 2, &mut rng));
            net.set_channel_mask(0, NeuronMask::from_off(4, &[2]));
            match compact(&net, &params) {
                Ok(_) if ok => agree(&net, &params, 36, 3),
                Err(Error::Unsupported(m)) if !ok => assert!(m.contains("layer 1 (prelu)"), "{m}"),
                other => panic!("{slopes} slopes: {:?}", other.map(drop)),
            }
        }
    }

    /// A mask that is not a whole-channel mask at a run's end, and channels that reach the
    /// output, are refused.
    #[test]
    fn refusals() {
        let mut params = ParamStore::new();
        let mut rng = Rng::seed(3);
        let mut net = Sequential::new()
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(1, 2, (3, 3), (2, 2)).unwrap(),
                &mut rng,
            ))
            .push(Relu)
            .push(Flatten)
            .push(Linear::new(&mut params, 8, 4, &mut rng))
            .push(Relu)
            .push(Linear::new(&mut params, 4, 2, &mut rng));
        net.set_mask(2, NeuronMask::from_off(8, &[0]));
        assert!(
            matches!(compact(&net, &params), Err(Error::Unsupported(m)) if m.contains("not a whole-channel mask"))
        );
        net.clear_mask(2);
        // A neuron mask on the last Linear: its neurons are the model's outputs.
        net.set_neuron_mask(5, NeuronMask::from_off(2, &[1]));
        assert!(
            matches!(compact(&net, &params), Err(Error::Unsupported(m)) if m.contains("no Linear layer follows it"))
        );
        let mut params = ParamStore::new();
        let mut last = Sequential::new()
            .push(Conv2d::new(
                &mut params,
                Conv2dGeom::new(1, 2, (3, 3), (2, 2)).unwrap(),
                &mut rng,
            ))
            .push(Relu);
        last.set_channel_mask(0, NeuronMask::from_off(2, &[0]));
        assert!(
            matches!(compact(&last, &params), Err(Error::Unsupported(m)) if m.contains("the model's output"))
        );
    }
}
