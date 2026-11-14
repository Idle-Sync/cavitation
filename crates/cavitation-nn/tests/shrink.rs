//! Tests of the shrinking tools.
//!
//! - `shrink::compact`: a masked network rebuilt without its off neurons gives the masked
//!   network's outputs, bit for bit on the kernel levels whose sums do not depend on an
//!   output's position and within the documented rounding bound on the others; the masks
//!   and layers it cannot compact are refused with a message that says what to do.
//! - `shrink::ablate`: the ablation sweep, after the compaction tests.
//! - `shrink::shrink`: the shrink tool, then its `fold_means` option.
//! - `shrink::train_found_size`: the from-scratch comparison, at the end.
//!
//! The kernel level is the one `CAVITATION_KERNEL` sets for the test process, except in
//! `matvec_drops_zero_terms_at_every_level`, which names each level it runs.

use cavitation_core::kernels::matvec::matvec_f32_with;
use cavitation_core::kernels::{Level, MatRef, supported};
use cavitation_core::{Error, Graph, ParamId, ParamStore, Real, Rng, Tape, Tensor, Var};
use cavitation_nn::shrink::{
    Ablation, Criterion, FromScratch, RankContext, Ranking, RetrainConfig, ShrinkConfig,
    ShrinkStep, ShrinkWarning, Validation, ablate, ablate_controlled, ablate_with, activity,
    compact, shrink, train_found_size,
};
use cavitation_nn::{
    Activation, ActivationLayer, Dataset, Evaluator, Flow, InferenceModel, Init, Layer, Linear,
    Loss, NeuronMask, Prelu, Relu, Sequential, Sigmoid, Softmax, Tanh, Targets, Trace, load, save,
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
                    (Targets::Classes(_), Some(b), Some(m)) => {
                        ((b - m) * 90.0).round() * 100.0 / 90.0
                    }
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

/// `ablate_controlled` ends the sweep at the first `Flow::Stop`, after that neuron, and ranks
/// the neurons measured so far exactly as the full sweep measured them.
#[test]
fn ablate_controlled_stops_where_asked() {
    let mut params = ParamStore::new();
    let net = small_network(&mut params);
    let data = value_data();
    let full = ablate(&net, &params, &data, &[0, 2], Loss::Mse, 32).unwrap();
    let mut calls = 0;
    let part = ablate_controlled(&net, &params, &data, &[0, 2], Loss::Mse, 32, &mut |_| {
        calls += 1;
        if calls == 4 {
            Flow::Stop
        } else {
            Flow::Continue
        }
    })
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(part.len(), 4);
    assert!(part.windows(2).all(|w| w[0].drop >= w[1].drop));
    for a in &part {
        assert_eq!((a.linear, a.neuron < 4), (0, true));
        let same = full
            .iter()
            .find(|f| (f.linear, f.neuron) == (a.linear, a.neuron))
            .unwrap();
        assert_eq!(a.drop.to_bits(), same.drop.to_bits());
    }
    // `Flow::Continue` throughout is the whole sweep.
    let all = ablate_controlled(&net, &params, &data, &[0, 2], Loss::Mse, 32, &mut |_| {
        Flow::Continue
    })
    .unwrap();
    assert_eq!(all, full);
}

/// Neurons off in a mask on the `Linear` itself (not at the neuron layer) are already off:
/// no entries, and they stay off, as the other neurons' measurements show.
#[test]
fn masks_anywhere_in_the_run_are_honoured() {
    let mut params = ParamStore::new();
    let mut net = small_network(&mut params);
    net.set_mask(0, NeuronMask::from_off(10, &[1, 4]));
    let data = class_data();
    let all = ablate(&net, &params, &data, &[0], Loss::CrossEntropy, 32).unwrap();
    assert_eq!(all.len(), 8);
    assert!(all.iter().all(|a| a.neuron != 1 && a.neuron != 4));
    let off: Vec<usize> = net.mask(0).unwrap().off().collect();
    assert_eq!(off, [1, 4]);
    assert!(net.neuron_mask(0).is_none());
    // Each entry is measured with 1 and 4 still off.
    for a in &all {
        let mut masked = net.clone();
        masked.set_mask(1, NeuronMask::from_off(10, &[a.neuron]));
        let want = Evaluator::new(&masked, &params, Loss::CrossEntropy, 32)
            .unwrap()
            .evaluate(&data)
            .unwrap();
        assert_eq!(a.evaluation, want);
    }
}

/// A negative baseline loss (KL divergence on small targets) gives finite drops whose sign
/// follows the loss: a higher loss is a positive drop.
#[test]
fn negative_baseline_loss_gives_finite_drops() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(8);
    let net = Sequential::new()
        .push(linear(&mut params, 4, 6, &mut rng))
        .push(Relu)
        .push(linear(&mut params, 6, 2, &mut rng));
    let x = inputs(40, 4, 9);
    let data = Dataset {
        x: Tensor::from_vec(x, &[40, 4]).unwrap(),
        targets: Targets::Values(Tensor::from_vec(vec![0.1f32; 80], &[40, 2]).unwrap()),
    };
    let base = Evaluator::new(&net, &params, Loss::KlDiv, 16)
        .unwrap()
        .evaluate(&data)
        .unwrap();
    assert!(base.loss < 0.0, "baseline {}", base.loss);
    let all = ablate(&net, &params, &data, &[0], Loss::KlDiv, 16).unwrap();
    assert_eq!(all.len(), 6);
    for a in &all {
        assert!(a.drop.is_finite());
        assert_eq!(a.drop, (a.evaluation.loss - base.loss) / base.loss.abs());
        assert_eq!(a.drop > 0.0, a.evaluation.loss > base.loss);
    }
    assert!(all.iter().any(|a| a.drop < 0.0) && all.iter().any(|a| a.drop > 0.0));
}

// The shrink tool.

/// `rows` seeded rows of `features` inputs in three classes: row `r` is in class `r % 3`
/// and lies around that class's centre (the same three centres for every seed), with
/// normal noise of standard deviation 0.8 on each input.
fn clusters(rows: usize, features: usize, seed: u64) -> Dataset {
    let mut at = Rng::seed(1000 + features as u64);
    let centres: Vec<f64> = (0..3 * features).map(|_| at.range(-2.0, 2.0)).collect();
    let mut rng = Rng::seed(seed);
    let mut x = Vec::with_capacity(rows * features);
    for r in 0..rows {
        let class = r % 3;
        for f in 0..features {
            x.push((centres[class * features + f] + 0.8 * rng.normal()) as f32);
        }
    }
    Dataset {
        x: Tensor::from_vec(x, &[rows, features]).unwrap(),
        targets: Targets::Classes((0..rows).map(|r| r % 3).collect()),
    }
}

/// A `features → width → ReLU → 3` network with seeded weights.
fn hidden_network(params: &mut ParamStore, features: usize, width: usize) -> Sequential {
    let mut rng = Rng::seed(3);
    Sequential::new()
        .push(linear(params, features, width, &mut rng))
        .push(Relu)
        .push(linear(params, width, 3, &mut rng))
}

/// The 8 → 32 → ReLU → 3 network trained for 20 epochs on 600 rows of `clusters`, with
/// those rows.
fn trained(params: &mut ParamStore) -> (Sequential, Dataset) {
    let net = hidden_network(params, 8, 32);
    let train = clusters(600, 8, 1);
    let config = cavitation_nn::FitConfig {
        epochs: 20,
        health: cavitation_nn::Health::quiet(),
        ..cavitation_nn::FitConfig::default()
    };
    cavitation_nn::fit(&net, params, &train, config).unwrap();
    (net, train)
}

/// A config that validates on `validation`, retrains one epoch per step and keeps every
/// checkpoint; the rest as the defaults.
fn quick(validation: Dataset) -> ShrinkConfig {
    ShrinkConfig {
        validation: Some(Validation::Given(validation)),
        retrain: RetrainConfig {
            epochs: 1,
            ..RetrainConfig::default()
        },
        keep_checkpoints: true,
        ..ShrinkConfig::default()
    }
}

/// A criterion every step meets: no accuracy drop exceeds 100 points.
const ALWAYS: Criterion = Criterion::AccuracyDrop { points: 100.0 };
/// A criterion no step meets: the accuracy would have to rise by more than 100 points.
const NEVER: Criterion = Criterion::AccuracyDrop { points: -101.0 };

/// The neurons a checkpoint's mask switches off on `Linear` `linear`.
fn off_in(step: &ShrinkStep, linear: usize) -> Vec<usize> {
    let (masks, _) = step.checkpoint.as_ref().expect("keep_checkpoints is on");
    let (_, mask) = masks.iter().find(|(l, _)| *l == linear).unwrap();
    mask.off().collect()
}

/// The default levels on a width of 256 keep 231, 205, 180, 154, 128, 103, 77, 64, 52, 39,
/// 26 and 13 neurons: twelve levels, 1,000 to 7,000 basis points by 1,000 then to 9,500 by
/// 500, and none lost to rounding.
#[test]
fn levels_follow_the_schedule() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 256);
    let train = clusters(64, 4, 2);
    let config = ShrinkConfig {
        criterion: Some(ALWAYS),
        patience: None,
        keep_checkpoints: false,
        ..quick(clusters(30, 4, 3))
    };
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    let kept: Vec<usize> = report.steps.iter().map(|s| s.kept[0]).collect();
    assert_eq!(kept, [231, 205, 180, 154, 128, 103, 77, 64, 52, 39, 26, 13]);
    let levels: Vec<u32> = report.steps.iter().map(|s| s.removed_bp).collect();
    assert_eq!(
        levels,
        [
            1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500, 8000, 8500, 9000, 9500
        ]
    );
    assert!(report.steps.iter().all(|s| s.checkpoint.is_none()));
    assert_eq!(report.chosen, Some(11));
    assert_eq!(net.neuron_mask(0).unwrap().count_on(), 13);
}

/// A 4 → 10 → ReLU → 3 network whose neurons 2 and 7 never fire (bias −1000, no incoming
/// weight) but have the largest outgoing weights, so the out-norm ranking alone would keep
/// them; with 32 rows of `clusters`.
fn dead_pair(params: &mut ParamStore) -> (Sequential, Dataset) {
    let net = hidden_network(params, 4, 10);
    let first = net.linear(0).unwrap();
    for dead in [2, 7] {
        params.value_mut(first.bias()).data_mut()[dead] = -1000.0;
        for w in &mut params.value_mut(first.weight()).data_mut()[dead * 4..dead * 4 + 4] {
            *w = 0.0;
        }
    }
    let last = net.linear(2).unwrap().weight();
    for row in params.value_mut(last).data_mut().chunks_mut(10) {
        row[2] = 10.0;
        row[7] = -10.0;
    }
    (net, clusters(32, 4, 4))
}

/// One level of 2,000 basis points on the dead pair's network.
fn one_level(validation: Dataset, dead_first: bool) -> ShrinkConfig {
    ShrinkConfig {
        step_bp: 2000,
        max_removed_bp: 2000,
        dead_first,
        criterion: Some(ALWAYS),
        ..quick(validation)
    }
}

/// With `dead_first`, the two neurons that never fire are the two removed at 20%, though
/// their outgoing weights are the largest.
#[test]
fn dead_neurons_go_first() {
    let mut params = ParamStore::new();
    let (mut net, train) = dead_pair(&mut params);
    let config = one_level(clusters(30, 4, 5), true);
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(report.steps.len(), 1);
    assert_eq!(report.steps[0].kept, [8]);
    assert_eq!(off_in(&report.steps[0], 0), [2, 7]);
}

/// With `dead_first` off, the dead neurons are ranked by out-norm like the rest: the two
/// live neurons with the smallest outgoing norms go, and the dead pair stays.
#[test]
fn dead_first_off_ranks_dead_by_norm() {
    let mut params = ParamStore::new();
    let (mut net, train) = dead_pair(&mut params);
    let w = params
        .value(net.linear(2).unwrap().weight())
        .data()
        .to_vec();
    let norm = |j: usize| {
        (0..3)
            .map(|r| f64::from(w[r * 10 + j]).powi(2))
            .sum::<f64>()
    };
    let mut live: Vec<usize> = (0..10).filter(|&j| j != 2 && j != 7).collect();
    live.sort_by(|&a, &b| norm(a).total_cmp(&norm(b)));
    let mut want = live[..2].to_vec();
    want.sort_unstable();
    let config = one_level(clusters(30, 4, 5), false);
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(off_in(&report.steps[0], 0), want);
}

/// With no patience, a run whose steps all miss goes through every level, chooses none,
/// and leaves the model unpruned: no mask, and the parameters as they were, bit for bit.
#[test]
fn patience_none_runs_every_level() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 16);
    let before = params.clone();
    let config = ShrinkConfig {
        criterion: Some(NEVER),
        patience: None,
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    assert_eq!(report.steps.len(), 12);
    assert!(report.steps.iter().all(|s| !s.met));
    assert_eq!(report.chosen, None);
    assert!(net.neuron_mask(0).is_none());
    assert_same_store(&net, &params, &before);
}

/// Asserts that every parameter of `net` holds the same bits in `got` and `want`.
fn assert_same_store(net: &Sequential, got: &ParamStore, want: &ParamStore) {
    for (name, id) in net.parameter_names() {
        assert_bits(got.value(id).data(), want.value(id).data(), &name);
    }
}

/// With the default patience of 2, two misses in a row end the run.
#[test]
fn patience_two_ends_after_two_misses() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 16);
    let config = ShrinkConfig {
        criterion: Some(NEVER),
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    assert_eq!(report.steps.len(), 2);
}

/// `Flow::Stop` from `on_step` after the third step ends the run there, keeping the best
/// step so far, and the callback sees each step as it is recorded.
#[test]
fn on_step_stop_keeps_best() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 16);
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let log = seen.clone();
    let config = ShrinkConfig {
        criterion: Some(ALWAYS),
        on_step: Some(Box::new(move |step: &ShrinkStep| {
            log.borrow_mut().push(step.removed_bp);
            if step.removed_bp == 3000 {
                Flow::Stop
            } else {
                Flow::Continue
            }
        })),
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    assert_eq!(*seen.borrow(), [1000, 2000, 3000]);
    assert_eq!(report.steps.len(), 3);
    assert_eq!(report.chosen, Some(2));
    assert_eq!(net.neuron_mask(0).unwrap().count_on(), 12);
}

/// The default config has no validation set: `InvalidData` naming `validation`, and the
/// model is not changed.
#[test]
fn missing_validation_is_invalid_data() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 16);
    let err = shrink(
        &mut net,
        &mut params,
        &clusters(48, 4, 7),
        ShrinkConfig::default(),
    )
    .unwrap_err();
    let Error::InvalidData(msg) = err else {
        panic!("expected InvalidData, got {err:?}");
    };
    assert!(msg.contains("validation"), "{msg}");
    assert!(msg.contains("must not have seen"), "{msg}");
    assert!(net.neuron_mask(0).is_none());
}

/// Value targets have no default criterion: `InvalidData` naming `criterion`; with
/// `LossIncrease` they shrink.
#[test]
fn value_targets_need_a_criterion() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 6, 12);
    let data = value_data();
    let (train, validation) = data.split_at(60);
    let retrain = RetrainConfig {
        epochs: 1,
        loss: Loss::Mse,
        ..RetrainConfig::default()
    };
    let config = ShrinkConfig {
        retrain,
        ..quick(validation.clone())
    };
    let err = shrink(&mut net, &mut params, &train, config).unwrap_err();
    let Error::InvalidData(msg) = err else {
        panic!("expected InvalidData, got {err:?}");
    };
    assert!(msg.contains("criterion"), "{msg}");
    let config = ShrinkConfig {
        retrain,
        criterion: Some(Criterion::LossIncrease { relative: 1e9 }),
        max_removed_bp: 3000,
        ..quick(validation.clone())
    };
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(report.steps.len(), 3);
    assert!(report.steps.iter().all(|s| s.met));
    // An accuracy criterion on value targets is refused too.
    let config = ShrinkConfig {
        retrain,
        criterion: Some(Criterion::AccuracyDrop { points: 1.0 }),
        ..quick(validation)
    };
    let err = shrink(&mut net, &mut params, &train, config).unwrap_err();
    assert!(err.to_string().contains("AccuracyDrop"), "{err}");
}

/// A validation set where one row is more than half the allowed drop warns: 150 rows give
/// 0.667 points a row against 1 point; 288 rows (0.347 points a row) do not.
#[test]
fn small_validation_warns() {
    let mut params = ParamStore::new();
    let net = hidden_network(&mut params, 4, 8);
    let run = |rows: usize| {
        let (mut net, mut params) = (net.clone(), params.clone());
        let config = ShrinkConfig {
            max_removed_bp: 1000,
            ..quick(clusters(rows, 4, 8))
        };
        shrink(&mut net, &mut params, &clusters(48, 4, 7), config)
            .unwrap()
            .warnings
    };
    let warnings = run(150);
    assert_eq!(warnings.len(), 1);
    let ShrinkWarning::SmallValidation {
        rows,
        row_points,
        threshold,
    } = warnings[0]
    else {
        panic!("expected SmallValidation, got {:?}", warnings[0]);
    };
    assert_eq!((rows, threshold), (150, 1.0));
    assert!((row_points - 100.0 / 150.0).abs() < 1e-12);
    assert!(warnings[0].to_string().contains("150"), "{}", warnings[0]);
    assert!(run(288).is_empty());
}

/// On return the model's masks are the chosen step's checkpoint masks and the parameters
/// its store, bit for bit.
#[test]
fn model_left_at_chosen_step() {
    let mut params = ParamStore::new();
    let (mut net, train) = trained(&mut params);
    let config = ShrinkConfig {
        patience: None,
        ..quick(clusters(200, 8, 9))
    };
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    let chosen = report.chosen.expect("a step meets the criterion");
    // A later step missed, so keeping the last step instead would show below.
    assert!(chosen + 1 < report.steps.len(), "chosen {chosen}");
    let (masks, store) = report.steps[chosen].checkpoint.as_ref().unwrap();
    assert_eq!(masks.len(), 1);
    for (linear, mask) in masks {
        assert_eq!(net.neuron_mask(*linear), Some(mask));
    }
    assert_same_store(&net, &params, store);
    // The last step's network differs, so the comparison above tells them apart.
    let (last_masks, last) = report.steps.last().unwrap().checkpoint.as_ref().unwrap();
    assert_ne!(last_masks, masks);
    let w = net.linear(0).unwrap().weight();
    assert_ne!(last.value(w).data(), params.value(w).data());
}

/// Removing 99.9% of 8 neurons keeps one: no level leaves a layer empty.
#[test]
fn at_least_one_neuron_kept() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 8);
    let config = ShrinkConfig {
        max_removed_bp: 9990,
        criterion: Some(ALWAYS),
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    let last = report.steps.last().unwrap();
    assert_eq!((last.removed_bp, last.kept[0]), (9990, 1));
    assert!(report.steps.iter().all(|s| s.kept[0] >= 1));
    assert_eq!(net.neuron_mask(0).unwrap().count_on(), 1);
}

/// `Ranking::Random` with one seed removes the same neurons on every run, and another seed
/// removes others.
#[test]
fn random_ranking_is_deterministic() {
    let mut params = ParamStore::new();
    let net = hidden_network(&mut params, 4, 32);
    let run = |seed: u64| {
        let (mut net, mut params) = (net.clone(), params.clone());
        let config = ShrinkConfig {
            ranking: Ranking::Random(seed),
            max_removed_bp: 5000,
            criterion: Some(ALWAYS),
            ..quick(clusters(30, 4, 6))
        };
        let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
        let offs: Vec<Vec<usize>> = report.steps.iter().map(|s| off_in(s, 0)).collect();
        let evaluations: Vec<_> = report.steps.iter().map(|s| s.validation).collect();
        (offs, evaluations)
    };
    let first = run(11);
    assert_eq!(first.0.len(), 5);
    assert_eq!(run(11), first);
    assert_ne!(run(12).0, first.0);
}

/// A model with no hidden `Linear` is refused, saying there is no hidden layer to shrink;
/// so is a listed layer that is not a hidden `Linear`.
#[test]
fn model_without_hidden_linear_refused() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(1);
    let lone = Sequential::new().push(linear(&mut params, 4, 3, &mut rng));
    let capped = lone.clone().push(Relu);
    for mut net in [lone, capped] {
        let err = shrink(
            &mut net,
            &mut params,
            &clusters(48, 4, 7),
            quick(clusters(30, 4, 6)),
        )
        .unwrap_err();
        let Error::InvalidData(msg) = err else {
            panic!("expected InvalidData, got {err:?}");
        };
        assert!(msg.contains("no hidden layer to shrink"), "{msg}");
    }
    let mut net = hidden_network(&mut params, 4, 8);
    for (bad, words) in [(1, "layer 1 is not a Linear"), (2, "last Linear")] {
        let config = ShrinkConfig {
            linears: vec![0, bad],
            ..quick(clusters(30, 4, 6))
        };
        let err = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap_err();
        assert!(err.to_string().contains(words), "{err}");
    }
}

/// A mask set with `set_mask` on the `Linear` itself (its neurons are read at the ReLU) is
/// refused: the tool masks neurons by `Linear`, at the ReLU.
#[test]
fn mask_inside_the_run_refused() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 8);
    net.set_mask(0, NeuronMask::from_off(8, &[1]));
    let err = shrink(
        &mut net,
        &mut params,
        &clusters(48, 4, 7),
        quick(clusters(30, 4, 6)),
    )
    .unwrap_err();
    let Error::Unsupported(msg) = err else {
        panic!("expected Unsupported, got {err:?}");
    };
    assert!(msg.contains("clear_mask(0)"), "{msg}");
}

/// Neurons the model already has off stay off and count as removed.
#[test]
fn existing_neuron_mask_counts_as_removed() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 10);
    net.set_neuron_mask(0, NeuronMask::from_off(10, &[0, 1, 2]));
    let config = ShrinkConfig {
        criterion: Some(ALWAYS),
        max_removed_bp: 5000,
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    let kept: Vec<usize> = report.steps.iter().map(|s| s.kept[0]).collect();
    // 9, 8 and 7 would be kept at 1,000 to 3,000; 7 are on already.
    assert_eq!(kept, [7, 7, 7, 6, 5]);
    for step in &report.steps {
        let off = off_in(step, 0);
        assert!(off.starts_with(&[0, 1, 2]), "{off:?}");
    }
}

/// `Validation::Carve` holds out the last `ceil(share · rows)` rows: the baseline is the
/// model measured on them, and shares that leave no row on either side are refused.
#[test]
fn carve_holds_out_the_last_rows() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 8);
    let train = clusters(100, 4, 7);
    let config = ShrinkConfig {
        validation: Some(Validation::Carve { share: 0.255 }),
        max_removed_bp: 1000,
        ..quick(clusters(1, 4, 1))
    };
    let held = train.subset(74..100);
    let want = cavitation_nn::evaluate(&net, &params, &held, Loss::CrossEntropy, 256).unwrap();
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(report.baseline, want);
    // 0.07 of 100 rows is 7 rows, though 0.07 · 100.0 is just above 7 in binary.
    let config = ShrinkConfig {
        validation: Some(Validation::Carve { share: 0.07 }),
        max_removed_bp: 1000,
        ..quick(clusters(1, 4, 1))
    };
    let seven = train.subset(93..100);
    let want = cavitation_nn::evaluate(&net, &params, &seven, Loss::CrossEntropy, 256).unwrap();
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(report.baseline, want);
    for share in [0.0, 1.0, -0.5, f64::NAN] {
        let config = ShrinkConfig {
            validation: Some(Validation::Carve { share }),
            ..ShrinkConfig::default()
        };
        let err = shrink(&mut net, &mut params, &train, config).unwrap_err();
        assert!(
            err.to_string().contains("must not have seen"),
            "{share}: {err}"
        );
    }
}

/// The 8 → 32 → ReLU → 3 network trained on 600 rows shrinks, with the defaults and a
/// given validation set of 200 rows, to a chosen step within 1 point of the baseline with
/// fewer than 32 neurons, whose parameter count is the compacted network's.
#[test]
fn shrink_end_to_end_small() {
    let mut params = ParamStore::new();
    let (mut net, train) = trained(&mut params);
    let config = ShrinkConfig {
        validation: Some(Validation::Given(clusters(200, 8, 9))),
        ..ShrinkConfig::default()
    };
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert!(report.warnings.is_empty());
    let chosen = report.chosen.expect("a step meets the criterion");
    let step = &report.steps[chosen];
    let (base, got) = (
        report.baseline.accuracy.unwrap(),
        step.validation.accuracy.unwrap(),
    );
    assert!(base - got <= 0.01 + 1e-12, "{base} → {got}");
    assert!(step.kept[0] < 32, "kept {}", step.kept[0]);
    assert!(step.checkpoint.is_none());
    let (small, small_params) = compact(&net, &params).unwrap();
    assert_eq!(small.linear(0).unwrap().outputs(), step.kept[0]);
    let count: usize = small
        .parameter_names()
        .iter()
        .map(|(_, id)| small_params.value(*id).data().len())
        .sum();
    assert_eq!(step.parameters, count);
}

/// Data that does not hold one target per row is `InvalidData` naming which set, not a
/// panic, for the training data (also when a validation set is carved from it) and for a
/// given validation set.
#[test]
fn bad_data_is_invalid_data() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 8);
    let mut short = clusters(30, 4, 6);
    short.targets = Targets::Classes(vec![0; 29]);
    let good = clusters(48, 4, 7);
    let cases = [
        (
            good.clone(),
            Validation::Given(short.clone()),
            "the validation set",
        ),
        (
            short.clone(),
            Validation::Given(good.clone()),
            "the training data",
        ),
        (short, Validation::Carve { share: 0.2 }, "the training data"),
    ];
    for (train, validation, words) in cases {
        let config = ShrinkConfig {
            validation: Some(validation),
            ..ShrinkConfig::default()
        };
        let err = shrink(&mut net, &mut params, &train, config).unwrap_err();
        let Error::InvalidData(msg) = err else {
            panic!("expected InvalidData, got {err:?}");
        };
        assert!(msg.contains(words) && msg.contains("29 labels"), "{msg}");
    }
}

/// `Validation::Carve` measures activity and retrains on the rows it does not hold out:
/// the ranking sees the activity of the first 74 rows, and the step's parameters are those
/// of `fit` on those rows with the step's masks, bit for bit.
#[test]
fn carve_retrains_on_the_rest() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 8);
    let (start_net, start) = (net.clone(), params.clone());
    let train = clusters(100, 4, 7);
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let log = seen.clone();
    let config = ShrinkConfig {
        validation: Some(Validation::Carve { share: 0.26 }),
        ranking: Ranking::Custom(Box::new(move |ctx| {
            log.borrow_mut().push(ctx.activity.clone());
            ctx.on.iter().map(|&j| j as f64).collect()
        })),
        step_bp: 2000,
        max_removed_bp: 2000,
        ..quick(clusters(1, 4, 1))
    };
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    let rest = train.subset(0..74);
    let want = activity(&start_net, &start, &rest, &[0], 256).unwrap();
    assert_eq!(*seen.borrow(), want);
    let (masks, store) = report.steps[0].checkpoint.as_ref().unwrap();
    let (mut by_hand, mut values) = (start_net.clone(), start.clone());
    by_hand.set_neuron_mask(0, masks[0].1.clone());
    let retrain = RetrainConfig::default();
    let config = cavitation_nn::FitConfig {
        epochs: 1,
        batch: retrain.batch,
        optimizer: retrain.optimizer,
        loss: retrain.loss,
        shuffle_seed: retrain.shuffle_seed,
        health: cavitation_nn::Health::quiet(),
        ..cavitation_nn::FitConfig::default()
    };
    cavitation_nn::fit(&by_hand, &mut values, &rest, config).unwrap();
    assert_same_store(&by_hand, store, &values);
}

/// A 4 → 10 → Tanh → 3 network whose neuron 4 is always about −1 (bias −5, no incoming
/// weight): it never fires but its output is far from 0, so it is not dead, and its mean
/// is the lowest while its mean |a| is the highest. Its outgoing weights are the largest.
fn tanh_network(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(3);
    let net = Sequential::new()
        .push(linear(params, 4, 10, &mut rng))
        .push(Tanh)
        .push(linear(params, 10, 3, &mut rng));
    let first = net.linear(0).unwrap();
    params.value_mut(first.bias()).data_mut()[4] = -5.0;
    for w in &mut params.value_mut(first.weight()).data_mut()[16..20] {
        *w = 0.0;
    }
    let last = net.linear(2).unwrap().weight();
    for row in params.value_mut(last).data_mut().chunks_mut(10) {
        row[4] = 10.0;
    }
    net
}

/// The two neurons with the lowest `key`, in increasing order.
fn lowest_two(key: impl Fn(usize) -> f64) -> Vec<usize> {
    let mut all: Vec<usize> = (0..10).collect();
    all.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
    let mut two = all[..2].to_vec();
    two.sort_unstable();
    two
}

/// On a Tanh network, a neuron that never fires but outputs about −1 is not dead: with
/// `dead_first` and the out-norm ranking it stays, and the two lowest out-norms go.
/// `Ranking::Activity` ranks by mean |a|, not the mean: the two lowest mean |a| go, and
/// the neuron with the lowest mean stays.
#[test]
fn tanh_neurons_rank_by_output_size() {
    let mut params = ParamStore::new();
    let net = tanh_network(&mut params);
    let train = clusters(32, 4, 4);
    let stats = activity(&net, &params, &train, &[0], 256)
        .unwrap()
        .remove(0);
    assert_eq!(stats.firing[4], 0.0);
    assert!(stats.mean_abs[4] > 0.99 && stats.mean[4] < -0.99);
    let w = params
        .value(net.linear(2).unwrap().weight())
        .data()
        .to_vec();
    let norm = |j: usize| {
        (0..3)
            .map(|r| f64::from(w[r * 10 + j]).powi(2))
            .sum::<f64>()
    };
    for (ranking, want) in [
        (Ranking::OutNorm, lowest_two(norm)),
        (Ranking::Activity, lowest_two(|j| stats.mean_abs[j])),
    ] {
        let (mut net, mut params) = (net.clone(), params.clone());
        let config = ShrinkConfig {
            ranking,
            ..one_level(clusters(30, 4, 5), true)
        };
        let report = shrink(&mut net, &mut params, &train, config).unwrap();
        let off = off_in(&report.steps[0], 0);
        assert_eq!(off, want);
        assert!(!off.contains(&4));
    }
}

/// A NaN score goes last whatever its sign bit: one computed at run time (`0.0 / 0.0`,
/// which has the sign bit set on common hardware) and `-NAN` are both kept, while the
/// lowest numbers go.
#[test]
fn nan_scores_go_last() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 10);
    let config = ShrinkConfig {
        ranking: Ranking::Custom(Box::new(|ctx| {
            let computed = std::hint::black_box(0.0_f64) / std::hint::black_box(0.0);
            ctx.on
                .iter()
                .map(|&j| match j {
                    0 => computed,
                    1 => -f64::NAN,
                    _ => j as f64,
                })
                .collect()
        })),
        step_bp: 5000,
        max_removed_bp: 5000,
        dead_first: false,
        criterion: Some(ALWAYS),
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    assert_eq!(off_in(&report.steps[0], 0), [2, 3, 4, 5, 6]);
}

/// With no retraining epochs each step only masks: its parameters are the ones given, bit
/// for bit.
#[test]
fn zero_epochs_leave_the_parameters() {
    let mut params = ParamStore::new();
    let mut net = hidden_network(&mut params, 4, 10);
    let before = params.clone();
    let config = ShrinkConfig {
        criterion: Some(ALWAYS),
        max_removed_bp: 3000,
        retrain: RetrainConfig {
            epochs: 0,
            ..RetrainConfig::default()
        },
        ..quick(clusters(30, 4, 6))
    };
    let report = shrink(&mut net, &mut params, &clusters(48, 4, 7), config).unwrap();
    assert_eq!(report.steps.len(), 3);
    for step in &report.steps {
        let (_, store) = step.checkpoint.as_ref().unwrap();
        assert_same_store(&net, store, &before);
    }
}

/// Runs `shrink` on `net` with a counter on `on_step`, and returns the result and the
/// number of steps the callback saw.
fn counted_shrink(
    net: &mut Sequential,
    params: &mut ParamStore,
    ranking: Ranking,
) -> (Result<cavitation_nn::shrink::ShrinkReport, Error>, usize) {
    let steps = std::rc::Rc::new(std::cell::Cell::new(0));
    let count = steps.clone();
    let config = ShrinkConfig {
        ranking,
        criterion: Some(ALWAYS),
        max_removed_bp: 3000,
        on_step: Some(Box::new(move |_: &ShrinkStep| {
            count.set(count.get() + 1);
            Flow::Continue
        })),
        ..quick(clusters(30, 4, 6))
    };
    let result = shrink(net, params, &clusters(48, 4, 7), config);
    (result, steps.get())
}

/// Neurons that reach a `Softmax` or a `Glu` before the next `Linear` are not removed by a
/// mask (a softmax still counts a 0 input, a GLU still passes half of a value whose gate
/// is 0), so `shrink` refuses them before any training, naming the layer, for every
/// ranking; the model and parameters are not changed.
#[test]
fn softmax_and_glu_refused_before_training() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(12);
    let softmax = Sequential::new()
        .push(linear(&mut params, 4, 8, &mut rng))
        .push(Tanh)
        .push(Softmax)
        .push(linear(&mut params, 8, 3, &mut rng));
    let glu = Sequential::new()
        .push(linear(&mut params, 4, 16, &mut rng))
        .push(cavitation_nn::Glu {
            gate: cavitation_core::Gate::Sigmoid,
        })
        .push(linear(&mut params, 8, 3, &mut rng));
    for (net, words) in [(softmax, "layer 2 (softmax)"), (glu, "layer 1 (glu)")] {
        for ranking in [Ranking::Activity, Ranking::OutNorm, Ranking::Random(1)] {
            let (mut net, mut values) = (net.clone(), params.clone());
            let (result, steps) = counted_shrink(&mut net, &mut values, ranking);
            let err = result.unwrap_err();
            let Error::Unsupported(msg) = err else {
                panic!("expected Unsupported, got {err:?}");
            };
            assert!(msg.contains(words) && msg.contains("not a Linear"), "{msg}");
            assert_eq!(steps, 0);
            assert!(net.neuron_mask(0).is_none());
            assert_same_store(&net, &values, &params);
        }
    }
}

/// Where masking removes neurons but `compact` cannot rebuild the result, `shrink` runs,
/// warns at the start with `NotCompactable` naming the layer, and `compact` then refuses
/// the chosen network: a layer of your own with parameters (after the output layer), and
/// an element-wise layer of your own without parameters in a shrunk run, and one that is
/// both (warned once).
#[test]
fn not_compactable_runs_and_warns() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(13);
    let alpha = params.add(Tensor::from_vec(vec![1.5], &[1]).unwrap());
    let with_parameters = Sequential::new()
        .push(linear(&mut params, 4, 8, &mut rng))
        .push(Relu)
        .push(linear(&mut params, 8, 3, &mut rng))
        .push(ScaledTanh { alpha });
    let own_elementwise = Sequential::new()
        .push(linear(&mut params, 4, 8, &mut rng))
        .push(Relu)
        .push(PositionScale)
        .push(linear(&mut params, 8, 3, &mut rng));
    let inner = Prelu::new(&mut params, 8);
    let both = Sequential::new()
        .push(linear(&mut params, 4, 8, &mut rng))
        .push(PreluTimesTwo { inner })
        .push(linear(&mut params, 8, 3, &mut rng));
    for (mut net, layer, words) in [
        (with_parameters, 3, "scaled_tanh has parameters"),
        (own_elementwise, 2, "position_scale"),
        // Element-wise and with parameters: one warning, for the parameters.
        (both, 1, "prelu_times_two has parameters"),
    ] {
        let (result, steps) = counted_shrink(&mut net, &mut params.clone(), Ranking::OutNorm);
        let report = result.unwrap();
        assert_eq!(steps, 3);
        assert!(report.chosen.is_some());
        let not_compactable: Vec<_> = report
            .warnings
            .iter()
            .filter_map(|w| match w {
                ShrinkWarning::NotCompactable { layer, reason } => Some((*layer, reason)),
                _ => None,
            })
            .collect();
        assert_eq!(not_compactable.len(), 1, "{:?}", report.warnings);
        assert_eq!(not_compactable[0].0, layer);
        assert!(
            not_compactable[0].1.contains(words),
            "{}",
            not_compactable[0].1
        );
        let text = report.warnings[0].to_string();
        assert!(text.contains("compact cannot rebuild"), "{text}");
        assert!(matches!(compact(&net, &params), Err(Error::Unsupported(_))));
    }
}

/// Every network shape `shrink` accepts without a `NotCompactable` warning compacts after
/// the run, for each built-in ranking, and the chosen step's parameter count is the
/// compacted network's: ReLU, Tanh then Sigmoid, a per-neuron and a one-slope `Prelu`, no
/// activation, an activation of your own (run on the tape), and two hidden layers.
#[test]
fn every_accepted_shape_compacts() {
    let mut params = ParamStore::new();
    let mut rng = Rng::seed(14);
    let mut l = |i: usize, o: usize| linear(&mut params, i, o, &mut rng);
    let shapes: Vec<(&str, Sequential)> = vec![
        (
            "relu",
            Sequential::new().push(l(4, 10)).push(Relu).push(l(10, 3)),
        ),
        (
            "tanh then sigmoid",
            Sequential::new()
                .push(l(4, 10))
                .push(Tanh)
                .push(Sigmoid)
                .push(l(10, 3)),
        ),
        (
            "no activation",
            Sequential::new().push(l(4, 10)).push(l(10, 3)),
        ),
        (
            "own activation",
            Sequential::new()
                .push(l(4, 10))
                .push(ActivationLayer::new(MyRelu))
                .push(l(10, 3)),
        ),
        (
            "two hidden",
            Sequential::new()
                .push(l(4, 12))
                .push(Relu)
                .push(l(12, 10))
                .push(Relu)
                .push(l(10, 3)),
        ),
    ];
    let mut shapes = shapes;
    for slopes in [10, 1] {
        let first = linear(&mut params, 4, 10, &mut rng);
        let prelu = Prelu::new(&mut params, slopes);
        let last = linear(&mut params, 10, 3, &mut rng);
        let name = if slopes == 1 {
            "one-slope prelu"
        } else {
            "prelu"
        };
        shapes.push((name, Sequential::new().push(first).push(prelu).push(last)));
    }
    for (name, net) in shapes {
        for ranking in [Ranking::OutNorm, Ranking::Activity, Ranking::Random(2)] {
            let what = format!("{name}, {ranking:?}");
            let (mut net, mut values) = (net.clone(), params.clone());
            let (result, _) = counted_shrink(&mut net, &mut values, ranking);
            let report = result.unwrap();
            assert!(
                report
                    .warnings
                    .iter()
                    .all(|w| !matches!(w, ShrinkWarning::NotCompactable { .. })),
                "{what}: {:?}",
                report.warnings
            );
            let chosen = &report.steps[report.chosen.expect("ALWAYS is met")];
            let (small, small_values) = compact(&net, &values).expect(&what);
            let count: usize = small
                .parameter_names()
                .iter()
                .map(|(_, id)| small_values.value(*id).data().len())
                .sum();
            assert_eq!(chosen.parameters, count, "{what}");
        }
    }
}

// Folding removed neurons' means into the next bias.

/// A 4 → 8 → ReLU → 6 → ReLU → 3 network, seeded as [`network`] is, whose neuron 2 of
/// layer 0 and neuron 1 of layer 2 have no incoming weight and a positive bias, so their
/// outputs are 0.7 and 0.4 on every row.
fn constant_pair(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(21);
    let net = Sequential::new()
        .push(linear(params, 4, 8, &mut rng))
        .push(Relu)
        .push(linear(params, 8, 6, &mut rng))
        .push(Relu)
        .push(linear(params, 6, 3, &mut rng));
    for (layer, neuron, value) in [(0, 2, 0.7), (2, 1, 0.4)] {
        let l = net.linear(layer).unwrap();
        let inputs = l.inputs();
        params.value_mut(l.bias()).data_mut()[neuron] = value;
        for w in &mut params.value_mut(l.weight()).data_mut()[neuron * inputs..][..inputs] {
            *w = 0.0;
        }
    }
    net
}

/// One step of 2,000 bp with no retraining that removes exactly `neurons[k]` from the
/// `k`-th hidden `Linear` (layers 0 and 2), with `fold_means` as given.
fn remove_exactly(neurons: [usize; 2], fold_means: bool, validation: Dataset) -> ShrinkConfig {
    ShrinkConfig {
        step_bp: 2000,
        max_removed_bp: 2000,
        dead_first: false,
        fold_means,
        criterion: Some(ALWAYS),
        ranking: Ranking::Custom(Box::new(move |ctx: &RankContext<'_>| {
            let target = if ctx.linear == 0 {
                neurons[0]
            } else {
                neurons[1]
            };
            ctx.on
                .iter()
                .map(|&j| if j == target { 0.0 } else { 1.0 })
                .collect()
        })),
        retrain: RetrainConfig {
            epochs: 0,
            ..RetrainConfig::default()
        },
        ..quick(validation)
    }
}

/// `fold_means` is off by default, and the config's `Debug` shows it.
#[test]
fn fold_means_is_off_by_default() {
    let config = ShrinkConfig::default();
    assert!(!config.fold_means);
    assert!(format!("{config:?}").contains("fold_means: false"));
}

/// With no retraining, removing a neuron whose output is the same on every row from each of
/// two hidden layers, with `fold_means`, leaves the network's outputs as they were: within
/// the sum of the two networks' GEMM bounds (each is within its own of the exact result,
/// and the exact results differ only by the rounding of each folded bias, which the
/// folded network's bound covers). Without the fold, the outputs move by far more.
#[test]
fn folding_a_constant_neuron_keeps_the_outputs() {
    let x = inputs(64, 4, 22);
    let mut original_params = ParamStore::new();
    let original = constant_pair(&mut original_params);
    let want = tape_output(&original, &original_params, &x, 64);
    let want_bound = gemm_bound(&original, &original_params, &x, 64);
    let train = clusters(48, 4, 23);
    for fold in [true, false] {
        let (mut net, mut params) = (original.clone(), original_params.clone());
        let config = remove_exactly([2, 1], fold, clusters(30, 4, 24));
        let report = shrink(&mut net, &mut params, &train, config).unwrap();
        assert_eq!(report.steps[0].kept, [7, 5]);
        assert_eq!(off_in(&report.steps[0], 0), [2]);
        assert_eq!(off_in(&report.steps[0], 2), [1]);
        // The kept network is the step's checkpoint, fold included.
        let (_, store) = report.steps[0].checkpoint.as_ref().unwrap();
        assert_same_store(&net, &params, store);
        let got = tape_output(&net, &params, &x, 64);
        let bound: Vec<f64> = gemm_bound(&net, &params, &x, 64)
            .iter()
            .zip(&want_bound)
            .map(|(a, b)| a + b)
            .collect();
        if fold {
            assert_within(&got, &want, &bound, "folded");
        } else {
            let moved = got
                .iter()
                .zip(&want)
                .zip(&bound)
                .any(|((g, w), b)| (f64::from(*g) - f64::from(*w)).abs() > 100.0 * b);
            assert!(moved, "without the fold the outputs should move");
        }
    }
}

/// Each layer's means are measured after the earlier layers have lost their neurons and
/// folded their means: the change in the output layer's bias is `W[:, j] · mean_j` for the
/// neuron `j` removed from layer 2, with `mean_j` measured on the network with layer 0
/// already pruned and folded, which differs from its mean in the network as given.
#[test]
fn later_means_are_measured_after_earlier_folds() {
    let mut params = ParamStore::new();
    let original = network(&mut params);
    let before = params.clone();
    let train = clusters(80, 64, 25);
    let mut net = original.clone();
    // Remove 20% of each hidden layer, the lowest indices first: neurons 0 to 50 of layer
    // 0 (205 of 256 kept) and 0 to 24 of layer 2 (103 of 128 kept).
    let mut config = remove_exactly([0, 0], true, clusters(30, 64, 26));
    config.ranking = Ranking::Custom(Box::new(|ctx: &RankContext<'_>| {
        ctx.on.iter().map(|&j| j as f64).collect()
    }));
    let report = shrink(&mut net, &mut params, &train, config).unwrap();
    assert_eq!(report.steps[0].kept, [205, 103]);
    let removed: Vec<usize> = off_in(&report.steps[0], 2);
    assert_eq!(removed, (0..25).collect::<Vec<_>>());
    // The network as it stood when layer 2's means were measured: layer 0 pruned and
    // folded, layer 2 whole (its own fold only changed layer 4's bias).
    let mut at_measure = net.clone();
    at_measure.clear_neuron_mask(2);
    let after = activity(&at_measure, &params, &train, &[2], 256).unwrap();
    let given = activity(&original, &before, &train, &[2], 256).unwrap();
    let out = net.linear(4).unwrap();
    let (w, inputs) = (before.value(out.weight()).data(), out.inputs());
    let (old_b, new_b) = (
        before.value(out.bias()).data(),
        params.value(out.bias()).data(),
    );
    let mut differs = false;
    for i in 0..out.outputs() {
        let fold = |mean: &[f64]| -> f64 {
            removed
                .iter()
                .map(|&j| f64::from(w[i * inputs + j]) * mean[j])
                .sum()
        };
        let (want, stale) = (fold(&after[0].mean), fold(&given[0].mean));
        let got = f64::from(new_b[i]) - f64::from(old_b[i]);
        let tolerance = 1.2e-7 * f64::from(new_b[i]).abs().max(f64::from(old_b[i]).abs());
        assert!(
            (got - want).abs() <= tolerance,
            "output {i}: bias moved by {got}, the fold of the means after layer 0 is {want}"
        );
        differs |= (stale - want).abs() > 1e3 * tolerance;
    }
    assert!(differs, "layer 0's pruning should change layer 2's means");
}

/// Without retraining, removing 70% of the hidden layer by activity costs less when the
/// removed neurons' means are folded into the next bias: on the trained 8 → 32 → 3 network,
/// 10 neurons kept, the validation accuracy on 600 rows is 93.50% without the fold and
/// 95.33% with it (measured on the `sse4.2` level), and the loss is lower.
#[test]
fn folding_helps_without_retraining() {
    let mut measured = Vec::new();
    for fold in [false, true] {
        let mut params = ParamStore::new();
        let (mut net, train) = trained(&mut params);
        let config = ShrinkConfig {
            ranking: Ranking::Activity,
            step_bp: 7000,
            max_removed_bp: 7000,
            fold_means: fold,
            criterion: Some(ALWAYS),
            retrain: RetrainConfig {
                epochs: 0,
                ..RetrainConfig::default()
            },
            ..quick(clusters(600, 8, 27))
        };
        let report = shrink(&mut net, &mut params, &train, config).unwrap();
        assert_eq!(report.steps[0].kept, [10]);
        measured.push(report.steps[0].validation);
    }
    let (without, with) = (measured[0], measured[1]);
    assert!(with.loss < without.loss, "{with:?} against {without:?}");
    let right = |e: cavitation_nn::Evaluation| (e.accuracy.unwrap() * 600.0).round() as usize;
    assert!(
        right(with) >= right(without) + 6,
        "{with:?} against {without:?}"
    );
}

// Training the found size from scratch.

/// `train_found_size` measures the pruned network as `evaluate` does, and trains a fresh
/// network that is, bit for bit, the network of the kept widths built by hand with the same
/// starting values and seed and trained with `fit` on the same recipe; a network `compact`
/// refuses is refused with its error.
#[test]
fn train_found_size_matches_a_network_built_by_hand() {
    let mut params = ParamStore::new();
    let (mut net, train) = trained(&mut params);
    net.set_neuron_mask(0, NeuronMask::from_off(32, &chosen(32, 20, 3)));
    let test = clusters(150, 8, 28);
    let recipe = |epochs| cavitation_nn::FitConfig {
        epochs,
        health: cavitation_nn::Health::quiet(),
        ..cavitation_nn::FitConfig::default()
    };
    let scratch = FromScratch {
        weights: Init::HeUniform,
        seed: 5,
        fit: recipe(4),
        ..FromScratch::default()
    };
    let found = train_found_size(&net, &params, &train, &test, scratch).unwrap();

    let mut by_hand_params = ParamStore::new();
    let mut rng = Rng::seed(5);
    let by_hand = Sequential::new()
        .push(Linear::with_init(
            &mut by_hand_params,
            8,
            12,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        ))
        .push(Relu)
        .push(Linear::with_init(
            &mut by_hand_params,
            12,
            3,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        ));
    let history = cavitation_nn::fit(&by_hand, &mut by_hand_params, &train, recipe(4)).unwrap();
    assert_eq!(found.history.epochs.len(), 4);
    for (a, b) in found.history.epochs.iter().zip(&history.epochs) {
        assert_eq!(a.mean_loss.to_bits(), b.mean_loss.to_bits());
    }
    assert_eq!(found.model.linear(0).unwrap().outputs(), 12);
    assert!(found.model.neuron_mask(0).is_none());
    let names = found.model.parameter_names();
    for ((name, got), (want_name, want)) in names.iter().zip(by_hand.parameter_names()) {
        assert_eq!(*name, want_name);
        let (got, want) = (found.params.value(*got), by_hand_params.value(want));
        assert_bits(got.data(), want.data(), name);
    }
    let loss = Loss::CrossEntropy;
    let scratch_eval = cavitation_nn::evaluate(&by_hand, &by_hand_params, &test, loss, 256);
    assert_eq!(found.scratch, scratch_eval.unwrap());
    let pruned_eval = cavitation_nn::evaluate(&net, &params, &test, loss, 256).unwrap();
    assert_eq!(found.pruned, pruned_eval);

    // A layer of your own with parameters: compact's refusal.
    let mut rng = Rng::seed(29);
    let alpha = params.add(Tensor::from_vec(vec![1.5], &[1]).unwrap());
    let mut foreign = Sequential::new()
        .push(linear(&mut params, 8, 6, &mut rng))
        .push(Relu)
        .push(linear(&mut params, 6, 3, &mut rng))
        .push(ScaledTanh { alpha });
    foreign.set_neuron_mask(0, NeuronMask::from_off(6, &[1]));
    let err = train_found_size(&foreign, &params, &train, &test, FromScratch::default());
    assert!(matches!(err, Err(Error::Unsupported(_))));
}
