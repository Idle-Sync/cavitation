//! Tests for `Sequential`'s fusion of a batch normalisation with the ReLU after it: the fused
//! network trains to the same bits as the unfused one, and the pair is fused only where it
//! should be (no masks, no kept output, a plain ReLU, fusion on).

use cavitation_core::{Conv2dGeom, Graph, ParamStore, Rng, Tensor};
use cavitation_nn::{BatchNorm2d, Conv2d, Hooks, NeuronMask, Relu, Relu6, Sequential, Trace};

/// conv(1 → 3, 3 × 3 on 5 × 5) → BatchNorm2d → ReLU → conv(3 → 2, 1 × 1 on 3 × 3), with
/// batch norm's β set so that some outputs are positive and some clamped.
fn net(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(4);
    let bn = BatchNorm2d::new(params, 3, (3, 3));
    let beta = bn.bias().unwrap();
    params
        .value_mut(beta)
        .data_mut()
        .copy_from_slice(&[0.5, -0.25, 0.0]);
    Sequential::new()
        .push(Conv2d::new(
            params,
            Conv2dGeom::new(1, 3, (5, 5), (3, 3)).unwrap(),
            &mut rng,
        ))
        .push(bn)
        .push(Relu)
        .push(Conv2d::new(
            params,
            Conv2dGeom::new(3, 2, (3, 3), (1, 1)).unwrap(),
            &mut rng,
        ))
}

/// `n` values uniform in [-1, 1) from `Rng::seed(seed)`.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..n).map(|_| rng.uniform_f32() * 2.0 - 1.0).collect()
}

/// Two training steps of `model` with SGD by hand (`p -= 0.1 · g`) on a fixed batch and the
/// loss `Σ y · r`, then one evaluation pass: every parameter's and buffer's bits after the
/// steps, the gradients of the last step, and the evaluation outputs.
fn train(model: &Sequential, mut params: ParamStore) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<u32>) {
    let batch = 4;
    let x = Tensor::from_vec(values(batch * 25, 1), &[batch, 25]).unwrap();
    let r = Tensor::from_vec(values(batch * 18, 2), &[batch, 18]).unwrap();
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    let ids: Vec<_> = model.state_names().into_iter().map(|(_, id)| id).collect();
    let params_ids: Vec<_> = model
        .parameter_names()
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut grads = Vec::new();
    for step in 1..=2 {
        params.zero_grad();
        let mut tape = graph.training_tape(&params.values, 0, step);
        let xv = tape.input(x.view());
        let y = model.forward_traced(&mut tape, xv, &mut trace);
        let rv = tape.input(r.view());
        let yr = tape.mul(y, rv);
        let loss = tape.sum(yr);
        tape.backward(loss, &mut params.grads);
        drop(tape);
        graph.apply_buffer_updates(&mut params.values);
        grads = params_ids
            .iter()
            .map(|&id| bits(params.grad(id).data()))
            .collect();
        for &id in &params_ids {
            let g = params.grad(id).data().to_vec();
            for (p, g) in params.value_mut(id).data_mut().iter_mut().zip(g) {
                *p -= 0.1 * g;
            }
        }
    }
    let state = ids
        .iter()
        .map(|&id| bits(params.value(id).data()))
        .collect();
    let mut tape = graph.tape(&params.values);
    let xv = tape.input(x.view());
    let y = model.forward(&mut tape, xv);
    let out = bits(tape.value(y).data());
    (state, grads, out)
}

/// The fused network's parameters, buffers, gradients and evaluation outputs after two
/// steps are the same bits as with fusion off.
#[test]
fn fused_training_matches_unfused() {
    let mut params = ParamStore::new();
    let fused = net(&mut params);
    let unfused = fused.clone().fuse(false);
    let (a, b) = (train(&fused, params.clone()), train(&unfused, params));
    assert_eq!(a.0, b.0, "parameters and buffers");
    assert_eq!(a.1, b.1, "gradients");
    assert_eq!(a.2, b.2, "evaluation outputs");
}

/// Which pairs a forward pass fuses, read from the trace.
fn fused_layers(model: &Sequential, params: &ParamStore, trace: &mut Trace) -> Vec<bool> {
    let mut graph = Graph::new();
    let mut tape = graph.training_tape(&params.values, 0, 1);
    let x = tape.input(
        Tensor::from_vec(values(2 * 25, 3), &[2, 25])
            .unwrap()
            .view(),
    );
    model.forward_traced(&mut tape, x, trace);
    (0..trace.len()).map(|i| trace.fused(i)).collect()
}

/// The batch norm is fused with the plain ReLU after it, and its trace entries are the
/// fused node; with fusion off, a mask on either layer, a kept output or a ReLU6 instead,
/// nothing is fused.
#[test]
fn fusion_happens_only_where_it_should() {
    let mut params = ParamStore::new();
    let model = net(&mut params);
    let mut trace = Trace::new();
    assert_eq!(
        fused_layers(&model, &params, &mut trace),
        [false, true, false, false]
    );
    assert_eq!(trace.output(1), trace.output(2));
    assert_eq!(trace.input(2), trace.output(1));

    let none = [false; 4];
    let off = model.clone().fuse(false);
    assert_eq!(fused_layers(&off, &params, &mut Trace::new()), none);
    for layer in [1, 2] {
        let mut masked = model.clone();
        masked.set_mask(layer, NeuronMask::all_on(27));
        assert_eq!(fused_layers(&masked, &params, &mut Trace::new()), none);
    }
    let mut kept = Trace::new();
    kept.keep_output(1);
    assert_eq!(fused_layers(&model, &params, &mut kept), none);
    let mut all = Trace::new();
    all.keep_all_outputs();
    assert_eq!(fused_layers(&model, &params, &mut all), none);

    let mut params6 = ParamStore::new();
    let mut rng = Rng::seed(4);
    let relu6 = Sequential::new()
        .push(BatchNorm2d::new(&mut params6, 1, (5, 5)))
        .push(Relu6)
        .push(Conv2d::new(
            &mut params6,
            Conv2dGeom::new(1, 1, (5, 5), (1, 1)).unwrap(),
            &mut rng,
        ));
    assert_eq!(
        fused_layers(&relu6, &params6, &mut Trace::new()),
        [false; 3]
    );
}

/// A hook on a fused batch norm (or on the ReLU, whose input is the batch norm's output)
/// refuses to run on a trace that fused it, and runs once the hooks' layers are kept.
#[test]
fn hooks_need_their_layers_kept() {
    let mut params = ParamStore::new();
    let model = net(&mut params);
    for layer in [1, 2] {
        let mut hooks = Hooks::new();
        hooks.on_forward(layer, |_| {});
        let mut graph = Graph::new();
        let mut tape = graph.training_tape(&params.values, 0, 1);
        let x = tape.input(
            Tensor::from_vec(values(2 * 25, 3), &[2, 25])
                .unwrap()
                .view(),
        );
        let mut trace = Trace::new();
        model.forward_traced(&mut tape, x, &mut trace);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            hooks.run_forward(&tape, &trace, &model);
        }));
        let message = *refused.unwrap_err().downcast::<String>().unwrap();
        assert!(message.contains("fused layer 1"), "{message}");
        hooks.keep_outputs(&mut trace);
        model.forward_traced(&mut tape, x, &mut trace);
        hooks.run_forward(&tape, &trace, &model);
    }
}
