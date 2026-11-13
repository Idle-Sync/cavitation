//! Tests of the step timings: `Tape::backward_timed` with each layer's output as a mark, and
//! the forward time per layer that `Sequential::forward_traced` records when the trace's
//! timing is on. They assert only what holds on a loaded machine: the parts add up to no
//! more than the whole, and a layer of real work takes a measurable time.

use std::time::{Duration, Instant};

use cavitation_core::{Graph, ParamStore, Reduction, Rng, Tensor, Var};
use cavitation_nn::{Init, Linear, Relu, Sequential, Trace};

/// 64 → 256 → 256 → 10 with ReLU between, He uniform weights from `Rng::seed(4)`.
fn wide_net(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(4);
    let mut he = |inputs, outputs| {
        Linear::with_init(
            params,
            inputs,
            outputs,
            Init::HeUniform,
            Init::Zeros,
            &mut rng,
        )
    };
    Sequential::new()
        .push(he(64, 256))
        .push(Relu)
        .push(he(256, 256))
        .push(Relu)
        .push(he(256, 10))
}

/// A 256-row batch of 64 features in `[-1, 1)` and its labels, from `Rng::seed(5)`.
fn batch() -> (Tensor, Vec<usize>) {
    let mut rng = Rng::seed(5);
    let x = (0..256 * 64).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    let labels = (0..256).map(|_| rng.below(10) as usize).collect();
    (Tensor::from_vec(x, &[256, 64]).unwrap(), labels)
}

/// The marks of a per-layer timed pass: each layer's output, in order.
fn marks(trace: &Trace) -> Vec<Var> {
    (0..trace.len()).map(|i| trace.output(i)).collect()
}

/// With each layer's output as a mark, the times add up to no more than the whole backward
/// pass measured around the call, and every `Linear`'s entry is above zero.
#[test]
fn timed_marks_cover_the_pass() {
    let mut params = ParamStore::new();
    let net = wide_net(&mut params);
    let (x, labels) = batch();
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    let logits = net.forward_traced(&mut tape, input, &mut trace);
    let loss = tape.softmax_cross_entropy(logits, &labels, Reduction::Mean);
    let marks = marks(&trace);
    let mut times = vec![Duration::ZERO; marks.len() + 1];
    let start = Instant::now();
    tape.backward_timed(loss, &mut params.grads, &marks, &mut times);
    let total = start.elapsed();
    let sum: Duration = times.iter().sum();
    assert!(
        sum <= total,
        "times add up to {sum:?}, the pass took {total:?}"
    );
    for i in [0, 2, 4] {
        assert!(times[i] > Duration::ZERO, "Linear {i}: {times:?}");
    }
}

/// `backward_timed` adds to `times` instead of overwriting, so a caller can sum over steps.
#[test]
fn timed_pass_adds_to_times() {
    let mut params = ParamStore::new();
    let net = wide_net(&mut params);
    let (x, labels) = batch();
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    let mut times = vec![Duration::ZERO; net.len() + 1];
    let mut after_first = Vec::new();
    for step in 0..2 {
        let mut tape = graph.tape(&params.values);
        let input = tape.input(x.view());
        let logits = net.forward_traced(&mut tape, input, &mut trace);
        let loss = tape.softmax_cross_entropy(logits, &labels, Reduction::Mean);
        tape.backward_timed(loss, &mut params.grads, &marks(&trace), &mut times);
        if step == 0 {
            after_first = times.clone();
        }
    }
    for i in [0, 2, 4] {
        assert!(
            times[i] > after_first[i],
            "Linear {i}: {after_first:?} then {times:?}"
        );
    }
}

/// With timing on, each layer's forward time is recorded, the times add up to no more than
/// the forward pass measured around the call, and every `Linear`'s is above zero. With it
/// off, the default, every layer reads zero.
#[test]
fn forward_times_when_timing_is_on() {
    let mut params = ParamStore::new();
    let net = wide_net(&mut params);
    let (x, _) = batch();
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    trace.set_timing(true);
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    let start = Instant::now();
    net.forward_traced(&mut tape, input, &mut trace);
    let total = start.elapsed();
    let sum: Duration = (0..trace.len()).map(|i| trace.forward_time(i)).sum();
    assert!(
        sum <= total,
        "times add up to {sum:?}, the pass took {total:?}"
    );
    for i in [0, 2, 4] {
        assert!(trace.forward_time(i) > Duration::ZERO, "Linear {i}");
    }
    drop(tape);

    trace.set_timing(false);
    let mut tape = graph.tape(&params.values);
    let input = tape.input(x.view());
    net.forward_traced(&mut tape, input, &mut trace);
    assert_eq!(trace.len(), 5);
    for i in 0..trace.len() {
        assert_eq!(trace.forward_time(i), Duration::ZERO, "layer {i}");
    }
}

/// Asking for the forward time of a layer the trace did not record panics.
#[test]
#[should_panic(expected = "trace: layer 0 out of range for 0 layers")]
fn forward_time_out_of_range_panics() {
    Trace::new().forward_time(0);
}
