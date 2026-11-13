//! Tests of `shrink::ablate` (the ablation sweep, at the end) and of `shrink::compact`: a masked network rebuilt without its off neurons gives the
//! masked network's outputs (bit for bit on the kernel levels whose sums do not depend on an
//! output's position, within the documented rounding bound on the others), and the masks
//! and layers it cannot compact are refused with a message that says what to do. The kernel
//! level is the one `CAVITATION_KERNEL` sets for the test process, except in
//! `matvec_drops_zero_terms_at_every_level`, which names each level it runs.

use cavitation_core::kernels::matvec::matvec_f32_with;
use cavitation_core::kernels::{Level, MatRef, supported};
use cavitation_core::{Error, Graph, ParamId, ParamStore, Real, Rng, Tape, Tensor, Var};
use cavitation_nn::shrink::{Ablation, ablate, ablate_with, compact};
use cavitation_nn::{
    Activation, ActivationLayer, Dataset, Evaluator, InferenceModel, Layer, Linear, Loss,
    NeuronMask, Prelu, Relu, Sequential, Sigmoid, Softmax, Tanh, Targets, Trace, load, save,
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

/// Whether compaction is exact at `level`: whether its matvec and GEMM compute every output
/// with the same steps whatever the output's position and the matrix's width, a sum in
/// input order from +0 with the multiply and the add rounded separately. The kernel docs
/// describe this for the reference loops and SSE4.2. AVX2 and NEON fuse the multiply and
/// add in the vector part of the matvec but not in its last few columns, and Accelerate's
/// GEMM keeps its own order, so removing neurons can change an output's rounding there.
fn compaction_is_exact(level: Level) -> bool {
    matches!(level, Level::Reference | Level::Sse42)
}

/// Asserts that `got`, from the compacted network, matches `want`, from the masked one, at
/// the current kernel level: bit for bit where [`compaction_is_exact`], and elsewhere within
/// twice `bound`, the masked network's rounding bound (each network is within it of the
/// exact in-order result, which is the same for both). Names `what` and the level.
fn assert_compacted(got: &[f32], want: &[f32], bound: &[f64], what: &str) {
    let level = cavitation_core::kernels::level();
    let what = format!("{what}, level {level}");
    if compaction_is_exact(level) {
        assert_bits(got, want, &what);
    } else {
        let twice: Vec<f64> = bound.iter().map(|b| 2.0 * b).collect();
        assert_within(got, want, &twice, &what);
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

/// Runs each row of `x` through `infer` on the masked and the compacted network and checks
/// them with [`assert_compacted`] against the masked network's bound, naming `what`.
fn check_infer(
    (net, params): (&Sequential, &ParamStore),
    (small, small_params): (&Sequential, &ParamStore),
    x: &[f32],
    what: &str,
) {
    let width = net
        .linear(0)
        .expect("the networks start with a Linear")
        .inputs();
    let rows = x.len() / width;
    let bound = gemm_bound(net, params, x, rows);
    let mut masked = InferenceModel::compile(net, params).unwrap();
    let mut compacted = InferenceModel::compile(small, small_params).unwrap();
    let outputs = masked.output_width();
    for (r, row) in x.chunks_exact(width).enumerate() {
        let want = masked.infer(row).to_vec();
        let got = compacted.infer(row);
        let b = &bound[r * outputs..(r + 1) * outputs];
        assert_compacted(got, &want, b, &format!("{what}, infer, input {r}"));
    }
}

/// Through `infer`, the compacted network gives the masked network's outputs on 360 seeded
/// inputs: bit for bit at the levels where compaction is exact, within the matvec bound
/// elsewhere.
#[test]
fn compact_matches_masked_bit_for_bit_through_infer() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    let x = inputs(ROWS, 64, 5);
    check_infer((&net, &params), (&small, &small_params), &x, "80% off");
}

/// Through `infer_batch` at batch 360 and through the tape, the compacted network is within
/// the GEMM bound of the masked one at every level, and bit for bit where compaction is
/// exact, since every summed length here is at most 256 (one block of the GEMM).
#[test]
fn compact_matches_within_gemm_bound_batched() {
    let mut params = ParamStore::new();
    let net = masked_network(&mut params);
    let (small, small_params) = compact(&net, &params).unwrap();
    let x = inputs(ROWS, 64, 6);
    let bound = gemm_bound(&net, &params, &x, ROWS);

    let mut masked = InferenceModel::compile(&net, &params).unwrap();
    let mut compacted = InferenceModel::compile(&small, &small_params).unwrap();
    let batch = Tensor::from_vec(x.clone(), &[ROWS, 64]).unwrap();
    let want = masked.infer_batch(batch.view()).data().to_vec();
    let got = compacted.infer_batch(batch.view()).data().to_vec();
    assert_compacted(&got, &want, &bound, "infer_batch");

    let want = tape_output(&net, &params, &x, ROWS);
    let got = tape_output(&small, &small_params, &x, ROWS);
    assert_compacted(&got, &want, &bound, "tape");
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
/// shared `Prelu` keeps its one slope; the outputs match through `infer`, as
/// [`assert_compacted`] checks.
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

    let x = inputs(50, 4, 9);
    check_infer((&net, &params), (&small, &small_params), &x, "prelu");
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
        msg.contains("layer 0")
            && msg.contains("clear_mask(0) and")
            && msg.contains("set_neuron_mask(0, mask)"),
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
/// matches the masked network through `infer`, as [`assert_compacted`] checks.
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
    let x = inputs(40, 64, 10);
    check_infer(
        (&net, &params),
        (&small, &small_params),
        &x,
        "mask on the linear",
    );
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
    /// Returns `"scaled_tanh"`.
    fn name(&self) -> &str {
        "scaled_tanh"
    }
    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    /// Records `tanh(x) · alpha`.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let t = tape.tanh(x);
        let alpha = tape.param(self.alpha);
        tape.mul(t, alpha)
    }
    /// Returns `alpha`.
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

/// An element-wise layer without parameters that scales position `j` by `j + 1`: it keeps
/// the `is_elementwise` promise, maps 0 to 0, and treats each position differently.
struct PositionScale;

impl<T: Real> Layer<T> for PositionScale {
    /// Returns `"position_scale"`.
    fn name(&self) -> &str {
        "position_scale"
    }
    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    /// Records `x · [1, 2, …, width]`.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let width = *tape.shape(x).dims().last().expect("a matrix input");
        let scale: Vec<T> = (1..=width).map(|j| T::from_f64(j as f64)).collect();
        let scale = Tensor::from_vec(scale, &[1, width]).unwrap();
        let s = tape.input(scale.view());
        tape.mul(x, s)
    }
    /// None.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        Vec::new()
    }
    /// `true`: each output depends only on the input at the same position.
    fn is_elementwise(&self) -> bool {
        true
    }
}

/// A layer that is element-wise but neither an activation nor a `Prelu` is refused, naming
/// it, when the `Linear` before it loses neurons, since moving a neuron to a lower position
/// could change what the layer does to it; with nothing removed it is shared as it is.
#[test]
fn position_dependent_layer_refused() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(8);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(PositionScale)
        .push(linear(&mut params, 4, 2, &mut rng));
    let (copy, _) = compact(&net, &params).unwrap();
    assert_eq!(copy.layer(1).name(), "position_scale");
    net.set_neuron_mask(0, NeuronMask::from_off(4, &[0]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("layer 1 (position_scale)") && msg.contains("every position alike"),
        "{msg}"
    );
}

/// The matvec of each level this CPU supports, named explicitly, on a masked layer (inputs
/// switched off as exact zeros, every output computed) and on its compacted form (those
/// inputs and four outputs removed, so outputs move between the vector part and the last
/// columns): bit for bit where [`compaction_is_exact`], within twice the matvec bound
/// `2·(k + 2)·1.19e-7·(Σᵢ |xᵢ|·|wᵢⱼ| + |bⱼ|)` (k = 20, the masked length) elsewhere.
#[test]
fn matvec_drops_zero_terms_at_every_level() {
    let (k, n) = (20, 13);
    let (off_inputs, off_outputs) = ([1, 5, 9, 10, 17], [0, 3, 6, 11]);
    let mut rng = Rng::seed(12);
    let w_t: Vec<f32> = (0..k * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    let bias: Vec<f32> = (0..n).map(|_| rng.range(-0.1, 0.1) as f32).collect();
    let mut x: Vec<f32> = (0..k).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    for &i in &off_inputs {
        x[i] = 0.0;
    }
    let rows: Vec<usize> = (0..k).filter(|i| !off_inputs.contains(i)).collect();
    let cols: Vec<usize> = (0..n).filter(|j| !off_outputs.contains(j)).collect();
    let small_w: Vec<f32> = rows
        .iter()
        .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
        .map(|(i, j)| w_t[i * n + j])
        .collect();
    let small_x: Vec<f32> = rows.iter().map(|&i| x[i]).collect();
    let small_bias: Vec<f32> = cols.iter().map(|&j| bias[j]).collect();
    let bound: Vec<f64> = cols
        .iter()
        .map(|&j| {
            let s: f64 = (0..k)
                .map(|i| f64::from(x[i]).abs() * f64::from(w_t[i * n + j]).abs())
                .sum();
            2.0 * 2.0 * (k as f64 + 2.0) * 1.19e-7 * (s + f64::from(bias[j]).abs())
        })
        .collect();
    let levels = [
        Level::Reference,
        Level::Sse42,
        Level::Avx2Fma,
        Level::Neon,
        Level::Accelerate,
    ];
    for level in levels.into_iter().filter(|&l| supported(l)) {
        let mut full = vec![0.0f32; n];
        let w = MatRef::new(&w_t, k, n, n, 1).unwrap();
        matvec_f32_with(level, w, &x, Some(&bias), false, &mut full);
        let want: Vec<f32> = cols.iter().map(|&j| full[j]).collect();
        let mut got = vec![0.0f32; cols.len()];
        let w = MatRef::new(&small_w, rows.len(), cols.len(), cols.len(), 1).unwrap();
        matvec_f32_with(level, w, &small_x, Some(&small_bias), false, &mut got);
        let what = format!("matvec, level {level}");
        if compaction_is_exact(level) {
            assert_bits(&got, &want, &what);
        } else {
            assert_within(&got, &want, &bound, &what);
        }
    }
}

/// A layer of your own that wraps a `Prelu` and doubles its output, and so has the
/// `Prelu`'s slopes as its parameters.
struct PreluTimesTwo {
    /// The wrapped `Prelu`.
    inner: Prelu,
}

impl<T: Real> Layer<T> for PreluTimesTwo {
    /// Returns `"prelu_times_two"`.
    fn name(&self) -> &str {
        "prelu_times_two"
    }
    /// The width is unchanged.
    fn output_width(&self, input_width: usize) -> usize {
        input_width
    }
    /// Records `2 · prelu(x)`.
    fn forward(&self, tape: &mut Tape<'_, T>, x: Var) -> Var {
        let y = self.inner.forward(tape, x);
        tape.scale(y, T::from_f64(2.0))
    }
    /// The wrapped `Prelu`'s slopes.
    fn parameters(&self) -> Vec<(&'static str, ParamId)> {
        vec![("weight", self.inner.weight())]
    }
    /// `true`: each output depends only on the input at the same position.
    fn is_elementwise(&self) -> bool {
        true
    }
}

/// A wrapper around a `Prelu` answers `as_prelu` with `None` and is refused as a layer with
/// parameters, naming it. That a layer of your own cannot answer `as_prelu` at all is
/// checked at compile time, by the `compile_fail` example on `as_prelu`: a test that runs
/// cannot hold such a layer.
#[test]
fn prelu_wrapper_is_refused_as_a_layer_with_parameters() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(9);
    let first = linear(&mut params, 3, 4, &mut rng);
    let inner = Prelu::new(&mut params, 4);
    let wrapper = PreluTimesTwo { inner };
    assert!((&wrapper as &dyn Layer<f32>).as_prelu().is_none());
    let mut net =
        Sequential::new()
            .push(first)
            .push(wrapper)
            .push(linear(&mut params, 4, 2, &mut rng));
    net.set_neuron_mask(0, NeuronMask::from_off(4, &[1]));
    let msg = refusal(&net, &params);
    assert!(msg.contains("layer 1 (prelu_times_two)"), "{msg}");
}

/// A mask with every neuron on removes nothing and is dropped, even on the output layer or
/// on a `Softmax`.
#[test]
fn all_on_masks_are_dropped() {
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_neuron_mask(4, NeuronMask::all_on(10));
    net.set_mask(1, NeuronMask::all_on(256));
    let (copy, copy_params) = compact(&net, &params).unwrap();
    assert_eq!(copy_params.len(), params.len());
    for i in 0..copy.len() {
        assert!(copy.mask(i).is_none(), "layer {i} keeps a mask");
    }

    let mut params = ParamStore::new();
    let mut rng = Rng::seed(10);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Softmax)
        .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(1, NeuronMask::all_on(4));
    assert!(compact(&net, &params).is_ok());
}

/// A neuron masked on a `Linear` that a `Sigmoid` follows is removed when a later mask in
/// the run (on the `Sigmoid`) switches it off too, since it then sends 0; a neuron only the
/// first mask switches off is still refused, naming it.
#[test]
fn a_later_mask_covers_an_earlier_one() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(11);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Sigmoid)
        .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(0, NeuronMask::from_off(4, &[2]));
    net.set_mask(1, NeuronMask::from_off(4, &[2]));
    let (small, small_params) = compact(&net, &params).unwrap();
    assert_eq!(small.linear(2).unwrap().inputs(), 3);
    let x = inputs(20, 3, 13);
    check_infer((&net, &params), (&small, &small_params), &x, "covered");

    net.set_mask(1, NeuronMask::from_off(4, &[1]));
    let msg = refusal(&net, &params);
    assert!(
        msg.contains("neuron 2") && msg.contains("clear_mask(0)"),
        "{msg}"
    );
}

/// The error of a `compact` that must fail with `Error::InvalidData`.
fn invalid(net: &Sequential, params: &ParamStore) -> String {
    match compact(net, params) {
        Err(Error::InvalidData(msg)) => msg,
        Err(other) => panic!("expected Error::InvalidData, got {other}"),
        Ok(_) => panic!("expected compact to fail"),
    }
}

/// A mask of the wrong width, a weight of the wrong shape in the store, a `Prelu` with the
/// wrong number of slopes and a next `Linear` with the wrong number of inputs are each
/// `Error::InvalidData`, naming the layer.
#[test]
fn misfitting_masks_and_values_are_invalid_data() {
    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_mask(1, NeuronMask::from_off(5, &[0]));
    let msg = invalid(&net, &params);
    assert!(msg.contains("layer 1") && msg.contains("width 5"), "{msg}");

    let mut params = ParamStore::new();
    let mut net = network(&mut params);
    net.set_neuron_mask(0, NeuronMask::from_off(256, &[0]));
    let weight = net.linear(2).unwrap().weight();
    *params.value_mut(weight) = Tensor::zeros(&[3, 3]);
    let msg = invalid(&net, &params);
    assert!(
        msg.contains("linear layer 2") && msg.contains("[3, 3]"),
        "{msg}"
    );

    let mut params = ParamStore::new();
    let mut rng = Rng::seed(14);
    let first = linear(&mut params, 3, 4, &mut rng);
    let prelu = Prelu::new(&mut params, 5);
    let mut net =
        Sequential::new()
            .push(first)
            .push(prelu)
            .push(linear(&mut params, 4, 2, &mut rng));
    net.set_mask(0, NeuronMask::from_off(4, &[0]));
    let msg = invalid(&net, &params);
    assert!(msg.contains("prelu layer 1 has 5 slopes"), "{msg}");

    let mut params = ParamStore::new();
    let mut rng = Rng::seed(15);
    let mut net = Sequential::new()
        .push(linear(&mut params, 3, 4, &mut rng))
        .push(Relu)
        .push(linear(&mut params, 5, 2, &mut rng));
    net.set_neuron_mask(0, NeuronMask::from_off(4, &[0]));
    let msg = invalid(&net, &params);
    assert!(msg.contains("linear layer 2 takes 5 inputs"), "{msg}");
}

/// `as_linear` and `as_prelu` answer on `dyn Layer` with and without `Send`, `Sync` or both
/// spelled out.
#[test]
fn downcasts_work_on_every_trait_object_form() {
    let mut params = ParamStore::new();
    let linear = Linear::new(&mut params, 2, 3, &mut Rng::seed(0));
    let prelu = Prelu::new(&mut params, 3);
    let plain: [Box<dyn Layer<f32>>; 2] = [Box::new(linear.clone()), Box::new(prelu)];
    let send: [Box<dyn Layer<f32> + Send>; 2] = [Box::new(linear.clone()), Box::new(prelu)];
    let sync: [Box<dyn Layer<f32> + Sync>; 2] = [Box::new(linear.clone()), Box::new(prelu)];
    let both: [std::sync::Arc<dyn Layer<f32> + Send + Sync>; 2] = [
        std::sync::Arc::new(linear.clone()),
        std::sync::Arc::new(prelu),
    ];
    assert_eq!(plain[0].as_linear().unwrap().outputs(), 3);
    assert_eq!(send[0].as_linear().unwrap().outputs(), 3);
    assert_eq!(sync[0].as_linear().unwrap().outputs(), 3);
    assert_eq!(both[0].as_linear().unwrap().outputs(), 3);
    for (what, linear_is_prelu, prelu_is_prelu, prelu_is_linear) in [
        (
            "plain",
            plain[0].as_prelu().is_some(),
            plain[1].as_prelu().is_some(),
            plain[1].as_linear().is_some(),
        ),
        (
            "send",
            send[0].as_prelu().is_some(),
            send[1].as_prelu().is_some(),
            send[1].as_linear().is_some(),
        ),
        (
            "sync",
            sync[0].as_prelu().is_some(),
            sync[1].as_prelu().is_some(),
            sync[1].as_linear().is_some(),
        ),
        (
            "both",
            both[0].as_prelu().is_some(),
            both[1].as_prelu().is_some(),
            both[1].as_linear().is_some(),
        ),
    ] {
        assert!(
            !linear_is_prelu && prelu_is_prelu && !prelu_is_linear,
            "{what}"
        );
    }
}

/// The 6 → 10 → ReLU → 8 → ReLU → 3 network, seeded, with a spread of weights.
fn small_network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(5);
    Sequential::new()
        .push(linear(params, 6, 10, &mut rng))
        .push(Relu)
        .push(linear(params, 10, 8, &mut rng))
        .push(Relu)
        .push(linear(params, 8, 3, &mut rng))
}

/// 90 seeded rows of 6 inputs with class labels from the sign pattern of the first two.
fn class_data() -> Dataset {
    let x = inputs(90, 6, 21);
    let labels = x
        .chunks(6)
        .map(|r| usize::from(r[0] > 0.0) + usize::from(r[1] > 0.0))
        .collect();
    Dataset {
        x: Tensor::from_vec(x, &[90, 6]).unwrap(),
        targets: Targets::Classes(labels),
    }
}

/// 90 seeded rows of 6 inputs with 3 value targets that follow the inputs.
fn value_data() -> Dataset {
    let x = inputs(90, 6, 22);
    let y: Vec<f32> = x
        .chunks(6)
        .flat_map(|r| [r[0] + r[1], r[2] - r[3], r[4] * 0.5])
        .collect();
    Dataset {
        x: Tensor::from_vec(x, &[90, 6]).unwrap(),
        targets: Targets::Values(Tensor::from_vec(y, &[90, 3]).unwrap()),
    }
}

/// The evaluation of `net` with the neuron `neuron` of `Linear` `linear` switched off on
/// top of the masks it has, through a plain `Evaluator`.
fn masked_evaluation(
    net: &Sequential,
    params: &ParamStore,
    data: &Dataset,
    loss: Loss,
    linear: usize,
    neuron: usize,
) -> cavitation_nn::Evaluation {
    let mut masked = net.clone();
    let mut mask = net
        .neuron_mask(linear)
        .cloned()
        .unwrap_or_else(|| NeuronMask::all_on(net.linear(linear).unwrap().outputs()));
    mask.set(neuron, false);
    masked.set_neuron_mask(linear, mask);
    Evaluator::new(&masked, params, loss, 32)
        .unwrap()
        .evaluate(data)
        .unwrap()
}

/// A neuron that never fires costs nothing: its ablation drops exactly 0.
#[test]
fn dead_neuron_ablation_drops_nothing() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let first = net.linear(0).unwrap();
    params.value_mut(first.bias()).data_mut()[3] = -1000.0;
    for w in &mut params.value_mut(first.weight()).data_mut()[3 * 6..4 * 6] {
        *w = 0.0;
    }
    for (data, loss) in [
        (class_data(), Loss::CrossEntropy),
        (value_data(), Loss::Mse),
    ] {
        let all = ablate(&net, &params, &data, &[0, 2], loss, 32).unwrap();
        let dead = all.iter().find(|a| (a.linear, a.neuron) == (0, 3)).unwrap();
        assert_eq!(dead.drop, 0.0);
        assert!(dead.drop.is_sign_positive());
        assert!(all.iter().any(|a| a.drop != 0.0));
    }
}

/// Each entry equals `evaluate` on the model with that one neuron masked, on class and on
/// value targets, with the model compiled and on the tape.
#[test]
fn ablation_matches_masked_evaluate() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let mut own = Sequential::new();
    for i in 0..net.len() {
        own = match net.linear(i) {
            Some(l) => own.push(l.clone()),
            None => own.push(ActivationLayer::new(MyRelu)),
        };
    }
    for model in [&net, &own] {
        for (data, loss) in [
            (class_data(), Loss::CrossEntropy),
            (value_data(), Loss::Mse),
        ] {
            let all = ablate(model, &params, &data, &[0, 2], loss, 32).unwrap();
            assert_eq!(all.len(), 10 + 8);
            let base = Evaluator::new(model, &params, loss, 32)
                .unwrap()
                .evaluate(&data)
                .unwrap();
            for a in &all {
                let want = masked_evaluation(model, &params, &data, loss, a.linear, a.neuron);
                assert_eq!(
                    a.evaluation, want,
                    "linear {} neuron {}",
                    a.linear, a.neuron
                );
                let drop = match (&data.targets, base.accuracy, a.evaluation.accuracy) {
                    (Targets::Classes(_), Some(b), Some(m)) => (b - m) * 100.0,
                    _ => (a.evaluation.loss - base.loss) / base.loss,
                };
                assert_eq!(a.drop, drop);
            }
        }
    }
}

/// ReLU written as an activation of your own, so the model has no inference form.
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

/// Neurons already off are skipped and stay off, in the model and in the other neurons'
/// measurements; the caller's model and masks are not changed.
#[test]
fn existing_masks_kept() {
    let mut params = ParamStore::new();
    let mut net = small_network(&mut params);
    net.set_neuron_mask(0, NeuronMask::from_off(10, &[2, 5]));
    let data = class_data();
    let all = ablate(&net, &params, &data, &[0, 2], Loss::CrossEntropy, 32).unwrap();
    assert_eq!(all.len(), 8 + 8);
    assert!(
        all.iter()
            .all(|a| a.linear != 0 || (a.neuron != 2 && a.neuron != 5))
    );
    let off: Vec<usize> = net.neuron_mask(0).unwrap().off().collect();
    assert_eq!(off, [2, 5]);
    assert!(net.neuron_mask(2).is_none());
    // Each entry is measured with 2 and 5 still off.
    for a in &all {
        let want = masked_evaluation(&net, &params, &data, Loss::CrossEntropy, a.linear, a.neuron);
        assert_eq!(a.evaluation, want);
    }
}

/// The list is sorted by drop, largest first, ties by (linear, neuron); class drops are in
/// accuracy points.
#[test]
fn sorted_by_drop() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let data = class_data();
    let all = ablate(&net, &params, &data, &[2, 0], Loss::CrossEntropy, 32).unwrap();
    for pair in all.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        assert!(
            a.drop > b.drop || (a.drop == b.drop && (a.linear, a.neuron) < (b.linear, b.neuron))
        );
    }
    let base = Evaluator::new(&net, &params, Loss::CrossEntropy, 32)
        .unwrap()
        .evaluate(&data)
        .unwrap()
        .accuracy
        .unwrap();
    let top = &all[0];
    assert!((top.drop - (base - top.evaluation.accuracy.unwrap()) * 100.0).abs() < 1e-9);
    // Row counts make the drops multiples of 100/90 points.
    assert!(all.iter().any(|a| a.drop != 0.0));
}

/// The output `Linear`, a layer that is not a `Linear` and an index past the end are
/// `InvalidData`, each naming the layer.
#[test]
fn output_linear_refused() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let data = class_data();
    for (bad, words) in [(4, "layer 4"), (1, "layer 1"), (9, "layer 9")] {
        let err = ablate(&net, &params, &data, &[0, bad], Loss::CrossEntropy, 32).unwrap_err();
        let Error::InvalidData(msg) = err else {
            panic!("expected InvalidData");
        };
        assert!(msg.contains(words), "{msg}");
    }
    let err = ablate(&net, &params, &data, &[4], Loss::CrossEntropy, 32).unwrap_err();
    assert!(err.to_string().contains("output"), "{err}");
}

/// `ablate_with` calls the closure once per neuron as it finishes, in sweep order, with the
/// entries `ablate_with` returns (unsorted order is the sweep order).
#[test]
fn ablate_with_reports_each_neuron() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let data = value_data();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut copies: Vec<Ablation> = Vec::new();
    let all = ablate_with(&net, &params, &data, &[0, 2], Loss::Mse, 32, &mut |a| {
        seen.push((a.linear, a.neuron));
        copies.push(*a);
    })
    .unwrap();
    let sweep: Vec<(usize, usize)> = (0..10)
        .map(|n| (0, n))
        .chain((0..8).map(|n| (2, n)))
        .collect();
    assert_eq!(seen, sweep);
    let mut sorted = copies;
    sorted.sort_by(|a, b| b.drop.total_cmp(&a.drop));
    let drops: Vec<f64> = all.iter().map(|a| a.drop).collect();
    let want: Vec<f64> = sorted.iter().map(|a| a.drop).collect();
    assert_eq!(drops, want);
    // A repeated Linear in the list is swept once; no Linear gives an empty result.
    let twice = ablate(&net, &params, &data, &[0, 0], Loss::Mse, 32).unwrap();
    assert_eq!(twice.len(), 10);
    assert!(
        ablate(&net, &params, &data, &[], Loss::Mse, 32)
            .unwrap()
            .is_empty()
    );
    assert!(ablate(&net, &params, &data, &[0], Loss::Mse, 0).is_err());
}
