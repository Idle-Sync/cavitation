//! Tests of `shrink::compact`: a masked network rebuilt without its off neurons gives the
//! masked network's outputs (bit for bit through `infer`, within the GEMM bound on the
//! batched paths), and the masks it cannot compact are refused with a message that says
//! what to do. The kernel level is the one `CAVITATION_KERNEL` sets for the test process.

use cavitation_core::{Error, Graph, ParamId, ParamStore, Real, Rng, Tape, Tensor, Var};
use cavitation_nn::shrink::compact;
use cavitation_nn::{
    InferenceModel, Layer, Linear, NeuronMask, Prelu, Relu, Sequential, Sigmoid, Softmax, Tanh,
    Trace, load, save,
};

/// The number of seeded inputs each comparison runs.
const ROWS: usize = 360;

/// A `Linear` of `inputs → outputs` with He-uniform weights and a seeded bias in
/// `[-0.1, 0.1)`, so that a wrongly kept or dropped bias shows.
fn linear(params: &mut ParamStore, inputs: usize, outputs: usize, rng: &mut Rng) -> Linear {
    let layer = Linear::with_init(
        params,
        inputs,
        outputs,
        cavitation_nn::Init::HeUniform,
        cavitation_nn::Init::Zeros,
        rng,
    );
    for b in params.value_mut(layer.bias()).data_mut() {
        *b = rng.range(-0.1, 0.1) as f32;
    }
    layer
}

/// The 64 → 256 → ReLU → 128 → ReLU → 10 network, seeded.
fn network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(13);
    Sequential::new()
        .push(linear(params, 64, 256, &mut rng))
        .push(Relu)
        .push(linear(params, 256, 128, &mut rng))
        .push(Relu)
        .push(linear(params, 128, 10, &mut rng))
}

/// `count` distinct neurons out of `width`, chosen with `seed`, in increasing order.
fn chosen(width: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut all: Vec<usize> = (0..width).collect();
    Rng::seed(seed).shuffle(&mut all);
    let mut some = all[..count].to_vec();
    some.sort_unstable();
    some
}

/// The network with 80% of each hidden layer switched off by `Linear`: 204 of 256 and 102
/// of 128.
fn masked_network(params: &mut ParamStore) -> Sequential {
    let mut net = network(params);
    net.set_neuron_mask(0, NeuronMask::from_off(256, &chosen(256, 204, 1)));
    net.set_neuron_mask(2, NeuronMask::from_off(128, &chosen(128, 102, 2)));
    net
}

/// `rows` seeded inputs of `width` values in `[-1, 1)`, row after row.
fn inputs(rows: usize, width: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..rows * width)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect()
}

/// Asserts `got` and `want` hold the same bits, naming `what` and the first element that
/// differs.
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} is {g:e}, the masked network gives {w:e}"
        );
    }
}

/// The tape's output of `net` on `rows` rows of `x` at once.
fn tape_output(net: &Sequential, params: &ParamStore, x: &[f32], rows: usize) -> Vec<f32> {
    let mut graph = Graph::new();
    let mut tape = graph.tape(&params.values);
    let input = Tensor::from_vec(x.to_vec(), &[rows, x.len() / rows]).unwrap();
    let xv = tape.input(input.view());
    let y = net.forward(&mut tape, xv);
    tape.value(y).data().to_vec()
}

/// The bound within which a batched path may differ from the masked network's tape output
/// on `rows` rows of `x`, for each output value: the v0.1 GEMM bound of each `Linear`,
/// `2·(k + 2)·1.19e-7·(Σᵢ |hᵢ|·|wᵢⱼ| + |bⱼ|)` with `k` the summed length of the masked
/// network and `h` the layer's input on its tape, plus the bound carried in from the layer
/// before, `Σᵢ boundᵢ·|wᵢⱼ|`. A mask sets the bound of its off neurons to 0; a ReLU carries
/// it through unchanged.
fn gemm_bound(net: &Sequential, params: &ParamStore, x: &[f32], rows: usize) -> Vec<f64> {
    let mut graph = Graph::new();
    let mut tape = graph.tape(&params.values);
    let input = Tensor::from_vec(x.to_vec(), &[rows, x.len() / rows]).unwrap();
    let xv = tape.input(input.view());
    let mut trace = Trace::new();
    net.forward_traced(&mut tape, xv, &mut trace);
    let mut bound = vec![0.0f64; x.len()];
    for i in 0..net.len() {
        if let Some(layer) = net.linear(i) {
            let (w, b) = (params.value(layer.weight()), params.value(layer.bias()));
            let (outputs, k) = (layer.outputs(), layer.inputs());
            let h = tape.value(trace.input(i));
            let (w, b, h) = (w.data(), b.data(), h.data());
            let mut next = vec![0.0f64; rows * outputs];
            for r in 0..rows {
                for j in 0..outputs {
                    let (mut s, mut carried) = (0.0f64, 0.0f64);
                    for p in 0..k {
                        let wv = f64::from(w[j * k + p]).abs();
                        s += f64::from(h[r * k + p]).abs() * wv;
                        carried += bound[r * k + p] * wv;
                    }
                    let own = 2.0 * (k as f64 + 2.0) * 1.19e-7 * (s + f64::from(b[j]).abs());
                    next[r * outputs + j] = own + carried;
                }
            }
            bound = next;
        }
        if let Some(mask) = net.mask(i) {
            let w = mask.width();
            for (x, b) in bound.iter_mut().enumerate() {
                if !mask.is_on(x % w) {
                    *b = 0.0;
                }
            }
        }
    }
    bound
}

/// Asserts `got` is within `bound` of `want` element by element, naming `what`.
fn assert_within(got: &[f32], want: &[f32], bound: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, ((&g, &w), &b)) in got.iter().zip(want).zip(bound).enumerate() {
        let diff = (f64::from(g) - f64::from(w)).abs();
        assert!(
            diff <= b,
            "{what}: element {i} is {g}, the masked network gives {w}, difference {diff:e} above {b:e}"
        );
    }
}

/// The error message of a refused `compact`, which must be `Error::Unsupported`.
fn refusal(net: &Sequential, params: &ParamStore) -> String {
    match compact(net, params) {
        Err(Error::Unsupported(msg)) => msg,
        Err(other) => panic!("expected Error::Unsupported, got {other}"),
        Ok(_) => panic!("expected compact to refuse"),
    }
}

/// Through `infer`, the compacted network gives the masked network's outputs bit for bit
/// on 360 seeded inputs, at the kernel level of this run.
#[test]
fn compact_matches_masked_bit_for_bit_through_infer() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    let x = inputs(ROWS, 64, 5);
    let level = cavitation_core::kernels::level();
    for (r, row) in x.as_chunks::<64>().0.iter().enumerate() {
        let want = masked.infer(row).to_vec();
        let got = compacted.infer(row);
        assert_bits(got, &want, &format!("infer, input {r}, level {level}"));
    }
}

/// Through `infer_batch` at batch 360 and through the tape, the compacted network is within
/// the GEMM bound of the masked one, and bit for bit here, where every summed length is at
/// most 256 (one block of the GEMM).
#[test]
fn compact_matches_within_gemm_bound_batched() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    let x = inputs(ROWS, 64, 6);
    let bound = gemm_bound(&net, &params, &x, ROWS);
    let level = cavitation_core::kernels::level();

    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    let batch = Tensor::from_vec(x.clone(), &[ROWS, 64]).unwrap();
    let want = masked.infer_batch(batch.view()).data().to_vec();
    let got = compacted.infer_batch(batch.view()).data().to_vec();
    assert_within(&got, &want, &bound, &format!("infer_batch, level {level}"));
    assert_bits(&got, &want, &format!("infer_batch, level {level}"));

    let want = tape_output(&net, &params, &x, ROWS);
    let got = tape_output(&small, &small_params, &x, ROWS);
    assert_within(&got, &want, &bound, &format!("tape, level {level}"));
    assert_bits(&got, &want, &format!("tape, level {level}"));
}

/// The compacted weights are the on rows of each masked `Linear` and the matching columns
/// of the next one: [52, 64], [26, 52] and [10, 26], with the values in place.
#[test]
fn compact_drops_rows_and_columns() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    assert_eq!(small.len(), net.len());
    let shapes: Vec<Vec<usize>> = small
        .linears()
        .map(|(_, l)| small_params.value(l.weight()).shape().dims().to_vec())
        .collect();
    assert_eq!(shapes, [vec![52, 64], vec![26, 52], vec![10, 26]]);
    assert_eq!(small_params.len(), 6);
    for i in 0..small.len() {
        assert!(small.mask(i).is_none(), "layer {i} keeps a mask");
    }

    let on0: Vec<usize> = net.neuron_mask(0).unwrap().on().collect();
    let on2: Vec<usize> = net.neuron_mask(2).unwrap().on().collect();
    let old = |i: usize| net.linear(i).unwrap();
    let new = |i: usize| small.linear(i).unwrap();
    let w = |p: &ParamStore, id: ParamId| p.value(id).data().to_vec();
    let (w0, w2, w4) = (
        w(&params, old(0).weight()),
        w(&params, old(2).weight()),
        w(&params, old(4).weight()),
    );
    let (b0, b2) = (w(&params, old(0).bias()), w(&params, old(2).bias()));
    let (n0, n2, n4) = (
        w(&small_params, new(0).weight()),
        w(&small_params, new(2).weight()),
        w(&small_params, new(4).weight()),
    );
    for (r, &j) in on0.iter().enumerate() {
        assert_eq!(
            n0[r * 64..r * 64 + 64],
            w0[j * 64..j * 64 + 64],
            "layer 0 row {r}"
        );
        assert_eq!(small_params.value(new(0).bias()).data()[r], b0[j]);
    }
    for (r, &j) in on2.iter().enumerate() {
        for (c, &i) in on0.iter().enumerate() {
            assert_eq!(n2[r * 52 + c], w2[j * 256 + i], "layer 2 ({r}, {c})");
        }
        assert_eq!(small_params.value(new(2).bias()).data()[r], b2[j]);
    }
    for r in 0..10 {
        for (c, &i) in on2.iter().enumerate() {
            assert_eq!(n4[r * 26 + c], w4[r * 128 + i], "layer 4 ({r}, {c})");
        }
    }
    assert_eq!(
        small_params.value(new(4).bias()).data(),
        params.value(old(4).bias()).data()
    );
}

/// A per-channel `Prelu` after a masked `Linear` keeps the slopes of the on neurons; a
/// shared `Prelu` keeps its one slope; the outputs match through `infer`.
#[test]
fn compact_prelu_per_channel() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(3);
    let first = linear(&mut params, 4, 6, &mut rng);
    let per_channel = Prelu::new(&mut params, 6);
    let slopes = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    params
        .value_mut(per_channel.weight())
        .data_mut()
        .copy_from_slice(&slopes);
    let second = linear(&mut params, 6, 5, &mut rng);
    let shared = Prelu::new(&mut params, 1);
    params.value_mut(shared.weight()).data_mut()[0] = 0.75;
    let third = linear(&mut params, 5, 2, &mut rng);
    let mut net = Sequential::new()
        .push(first)
        .push(per_channel)
        .push(second)
        .push(shared)
        .push(third);
    net.set_neuron_mask(0, NeuronMask::from_off(6, &[0, 3, 4]));
    net.set_neuron_mask(2, NeuronMask::from_off(5, &[1]));
    let (small, small_params) = compact(&net, &params).unwrap();

    let prelu = small.layer(1).as_prelu().expect("layer 1 is a Prelu");
    assert_eq!(small_params.value(prelu.weight()).data(), &[0.2, 0.3, 0.6]);
    let prelu = small.layer(3).as_prelu().expect("layer 3 is a Prelu");
    assert_eq!(small_params.value(prelu.weight()).data(), &[0.75]);
    assert_eq!(small.linear(2).unwrap().inputs(), 3);
    assert_eq!(small.linear(4).unwrap().inputs(), 4);

    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    for (r, row) in inputs(50, 4, 9).as_chunks::<4>().0.iter().enumerate() {
        let want = masked.infer(row).to_vec();
        assert_bits(compacted.infer(row), &want, &format!("input {r}"));
    }
}

/// A mask on the output layer is refused, naming the layer.
#[test]
fn mask_on_output_refused() {
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_neuron_mask(4, NeuronMask::from_off(10, &[3]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 4") && msg.contains("outputs") && msg.contains("clear_mask(4)"),
        "{msg}"
    );
}

/// A mask on a `Linear` that a `Sigmoid` follows sends `sigmoid(0) = 0.5`, not 0: refused,
/// naming layer 0 and `set_neuron_mask`.
#[test]
fn mask_on_linear_before_sigmoid_refused() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(4);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Sigmoid)
        .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(0, NeuronMask::from_off(4, &[2]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 0") && msg.contains("set_neuron_mask(0, mask)"),
        "{msg}"
    );
    assert!(msg.contains("layer 1 (sigmoid)"), "{msg}");
}

/// A mask on a `Tanh` that a `Sigmoid` follows is refused the same way.
#[test]
fn mask_on_tanh_then_sigmoid_refused() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(5);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Tanh)
        .push(Sigmoid)
        .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(1, NeuronMask::from_off(4, &[0]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 1 (tanh)")
            && msg.contains("layer 2 (sigmoid)")
            && msg.contains("set_neuron_mask(0, mask)")
            && msg.contains("layer 2, after every activation"),
        "{msg}"
    );
    // The same neurons masked after the sigmoid compact.
    net.clear_mask(1);
    net.set_neuron_mask(0, NeuronMask::from_off(4, &[0]));
    let (small, _) = compact(&net, &params).unwrap();
    assert_eq!(small.linear(3).unwrap().inputs(), 3);
}

/// A mask on a `Linear` itself compacts when the layers after it map 0 to 0 (a ReLU), and
/// matches the masked network through `infer`.
#[test]
fn mask_on_linear_before_relu_compacts() {
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_mask(0, NeuronMask::from_off(256, &chosen(256, 100, 7)));
    net.set_mask(1, NeuronMask::from_off(256, &chosen(256, 100, 8)));
    let (small, small_params) = compact(&net, &params).unwrap();
    let union = {
        let (a, b) = (net.mask(0).unwrap(), net.mask(1).unwrap());
        (0..256).filter(|&j| a.is_on(j) && b.is_on(j)).count()
    };
    assert_eq!(small.linear(0).unwrap().outputs(), union);
    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    for (r, row) in inputs(40, 64, 10).as_chunks::<64>().0.iter().enumerate() {
        let want = masked.infer(row).to_vec();
        assert_bits(compacted.infer(row), &want, &format!("input {r}"));
    }
}

/// A mask whose neurons reach a `Softmax` before the next `Linear` is refused, and so is a
/// `Linear` with every neuron off.
#[test]
fn masks_that_do_not_reach_a_linear_or_leave_nothing_refused() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(6);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Relu)
        .push(Softmax)
        .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(1, NeuronMask::from_off(4, &[1]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 2 (softmax)") && msg.contains("clear_mask(1)"),
        "{msg}"
    );
    net.clear_mask(1);
    net.set_mask(2, NeuronMask::from_off(4, &[1]));
    let msg = refusal(&net, &params);
    assert!(msg.contains("layer 2 (softmax)"), "{msg}");

    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_neuron_mask(2, NeuronMask::all_off(128));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("linear layer 2") && msg.contains("set_neuron(2"),
        "{msg}"
    );
}

/// `tanh(x)` scaled by a learnable `alpha`: a layer of your own with a parameter.
struct ScaledTanh {
    /// The scale.
    alpha: ParamId,
}

impl<T: Real> Layer<T> for ScaledTanh {
    fn name(&self) -> &str {
        "scaled_tanh"
    }
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let t = tape.tanh(x);
        let alpha = tape.param(self.alpha);
        tape.mul(t, alpha)
    }
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        vec![("alpha", self.alpha)]
    }
}

/// A layer of your own with parameters is refused, naming it, even with no mask.
#[test]
fn compact_refuses_user_layer_with_parameters() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(7);
    let first = linear(&mut params, 3, 4, &mut rng);
    let alpha = params.add(Tensor::from_vec(vec![1.5], &[1]).unwrap());
    let net = Sequential::new()
        .push(first)
        .push(ScaledTanh { alpha })
        .push(linear(&mut params, 4, 2, &mut rng));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 1 (scaled_tanh)") && msg.contains("parameters"),
        "{msg}"
    );
}

/// With no mask, the result is a copy: the same layers, shapes and values in a new store,
/// the layers without parameters shared, and the same outputs.
#[test]
fn unmasked_network_compacts_to_a_copy() {
    let mut params = ParamStore::new();
    let net = network(&mut params);
    let (copy, copy_params) = compact(&net, &params).unwrap();
    assert_eq!(copy.len(), net.len());
    assert_eq!(copy_params.len(), params.len());
    let names = |n: &Sequential| -> Vec<String> {
        n.parameter_names()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    };
    assert_eq!(names(&copy), names(&net));
    for ((_, a), (_, b)) in net.parameter_names().iter().zip(copy.parameter_names()) {
        assert_eq!(params.value(*a), copy_params.value(b));
    }
    for i in 0..net.len() {
        assert_eq!(copy.layer(i).name(), net.layer(i).name());
    }
    let x = inputs(ROWS, 64, 11);
    assert_bits(
        &tape_output(&copy, &copy_params, &x, ROWS),
        &tape_output(&net, &params, &x, ROWS),
        "copy",
    );
}

/// The compacted network saves and loads: its values come back bit for bit.
#[test]
fn compacted_saves_and_loads() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    let path = std::env::temp_dir().join(format!(
        "cavitation-compacted-{}.safetensors",
        std::process::id()
    ));
    save(&path, &small, &small_params).unwrap();
    let mut loaded = small_params.clone();
    for (_, id) in small.parameter_names() {
        loaded.value_mut(id).data_mut().fill(0.0);
    }
    load(&path, &small, &mut loaded).unwrap();
    std::fs::remove_file(&path).unwrap();
    for (name, id) in small.parameter_names() {
        let (a, b) = (small_params.value(id).data(), loaded.value(id).data());
        assert_bits(b, a, &name);
    }
}

/// A sum that could end at -0 cannot differ: each matvec sum starts from +0, and +0 plus
/// -0 is +0, so with a bias of -0 the masked and compacted outputs still hold the same
/// bits. Here every kept product is -0 (input 0 times a negative weight), every dropped one
/// +0, and the bias -0.
#[test]
fn negative_zero_sums_match() {
    let mut params = ParamStore::new();
    let first = Linear::from_values(
        &mut params,
        Tensor::from_vec(vec![0.0f32, 0.0, 0.0], &[3, 1]).unwrap(),
        Tensor::from_vec(vec![0.0f32, 0.0, 0.0], &[3]).unwrap(),
    )
    .unwrap();
    let second = Linear::from_values(
        &mut params,
        Tensor::from_vec(vec![-1.0f32, 2.0, -3.0], &[1, 3]).unwrap(),
        Tensor::from_vec(vec![-0.0f32], &[1]).unwrap(),
    )
    .unwrap();
    let mut net = Sequential::new().push(first).push(Relu).push(second);
    net.set_neuron_mask(0, NeuronMask::from_off(3, &[1]));
    let (small, small_params) = compact(&net, &params).unwrap();
    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    let want = masked.infer(&[1.0]).to_vec();
    assert_bits(compacted.infer(&[1.0]), &want, "negative zero");
    assert_eq!(want[0].to_bits(), 0.0f32.to_bits());
}

/// Compaction is exact for finite weights only: a masked neuron whose outgoing weight is
/// infinite gives `0 · ∞ = NaN` in the masked network, while the compacted network drops
/// that weight and gives the sum of the kept terms.
#[test]
fn infinite_weight_on_a_masked_neuron_differs() {
    let mut params = ParamStore::new();
    let first = Linear::from_values(
        &mut params,
        Tensor::from_vec(vec![1.0f32, 1.0, 1.0], &[3, 1]).unwrap(),
        Tensor::from_vec(vec![0.0f32, 0.0, 0.0], &[3]).unwrap(),
    )
    .unwrap();
    let second = Linear::from_values(
        &mut params,
        Tensor::from_vec(vec![1.0f32, f32::INFINITY, 2.0], &[1, 3]).unwrap(),
        Tensor::from_vec(vec![0.5f32], &[1]).unwrap(),
    )
    .unwrap();
    let mut net = Sequential::new().push(first).push(Relu).push(second);
    net.set_neuron_mask(0, NeuronMask::from_off(3, &[1]));
    let (small, small_params) = compact(&net, &params).unwrap();
    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    assert!(masked.infer(&[2.0])[0].is_nan());
    assert_eq!(compacted.infer(&[2.0]), &[6.5]);
}
