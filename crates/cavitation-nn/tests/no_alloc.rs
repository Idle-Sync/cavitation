//! Counts heap allocations in steady-state steps; any allocation after warm-up fails the run.
//!
//! This target has no test harness: `main` runs each scenario in turn on the main thread, so
//! nothing else allocates while a scenario is being counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cavitation_core::{Graph, ParamGrads, ParamStore, Reduction, Rng, Tensor, TensorView, Var};
use cavitation_nn::{
    Activation, ActivationLayer, Adam, AdamW, Batch, Batcher, Dataset, Evaluator, GradStats,
    Health, Hooks, InferenceModel, Init, Linear, Loss, NeuronMask, OnNonFinite, Optimizer, Relu,
    Sequential, Sgd, Targets, TargetsView, Trace, Trainer, Warning,
};

/// The number of allocations (including reallocations) made so far, from every thread.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting every allocation and reallocation in [`ALLOCATIONS`].
struct CountingAllocator;

// SAFETY: every method forwards its arguments unchanged to `System`, which implements
// `GlobalAlloc` correctly; counting touches only an atomic and never allocates.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract, passed on unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` and `layout` came from this allocator, which hands out `System`'s
        // blocks unchanged, and the caller upholds the rest of `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from this allocator, which hands out `System`'s
        // blocks unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Installs the counting allocator for the whole process.
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The number of allocations so far.
fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// A scenario: its name and a function returning the allocations made after warm-up.
type Scenario = (&'static str, fn() -> usize);

/// Steps each of the four optimisers 12 times on a store of a matrix, a vector and a
/// 300 by 200 matrix (60,000 elements: several chunks, so the update runs on the thread pool)
/// with fresh gradients each step, reusing one `GradStats`. Returns the allocations made
/// during steps 2 to 12 (the first step allocates the state and starts the pool's workers).
fn optimizers_steady_state() -> usize {
    let mut optimizers: Vec<Box<dyn Optimizer<f32>>> = vec![
        Box::new(Sgd::new(0.1)),
        Box::new(Sgd::new(0.1).with_momentum(0.9).with_weight_decay(0.01)),
        Box::new(Adam::new(0.01).with_weight_decay(0.1)),
        Box::new(AdamW::new(0.01)),
    ];
    let mut store = ParamStore::new();
    let w = store.add(Tensor::full(&[4, 3], 0.5f32));
    let b = store.add(Tensor::full(&[3], -0.5f32));
    let big = store.add(Tensor::full(&[300, 200], 0.1f32));
    let mut stats = GradStats::new();
    let mut before = 0;
    let mut sink = 0.0;
    for step in 1..=12 {
        if step == 2 {
            before = allocations();
        }
        for opt in &mut optimizers {
            store.zero_grad();
            store.grads.accumulate(w).0.data_mut().fill(0.25);
            store.grads.accumulate(b).0.data_mut().fill(-0.125);
            store.grads.accumulate(big).0.data_mut().fill(0.5);
            opt.step_with_stats(&mut store, &mut stats);
            sink += stats.norm(w);
        }
    }
    assert!(sink > 0.0);
    allocations() - before
}

/// Steps each of the four optimisers with `step` (the path a training loop uses) 12 times,
/// alternating fresh gradients with `zero_grad` and no backward pass (stale gradients), and
/// reads `last_stats` each time. Returns the allocations made during steps 2 to 12.
fn optimizers_step_steady_state() -> usize {
    let mut optimizers: Vec<Box<dyn Optimizer<f32>>> = vec![
        Box::new(Sgd::new(0.1)),
        Box::new(Sgd::new(0.1).with_momentum(0.9).with_weight_decay(0.01)),
        Box::new(Adam::new(0.01).with_weight_decay(0.1)),
        Box::new(AdamW::new(0.01)),
    ];
    let mut store = ParamStore::new();
    let w = store.add(Tensor::full(&[4, 3], 0.5f32));
    let b = store.add(Tensor::full(&[3], -0.5f32));
    let mut before = 0;
    let mut sink = 0.0;
    for step in 1..=12 {
        if step == 2 {
            before = allocations();
        }
        for opt in &mut optimizers {
            store.zero_grad();
            if step % 2 == 1 {
                store.grads.accumulate(w).0.data_mut().fill(0.25);
                store.grads.accumulate(b).0.data_mut().fill(-0.125);
            }
            opt.step(&mut store);
            sink += opt.last_stats().norm(w);
        }
    }
    assert!(sink > 0.0);
    allocations() - before
}

/// Runs the batcher over 1,437 rows at batch 128 (12 batches, the last of 29), for class
/// labels and for value targets: one warm-up epoch, then three epochs of `epoch`, `batches`
/// and `gather`. Returns the allocations made after the warm-up epoch.
fn gather_steady_state() -> usize {
    let n = 1437;
    let x: Vec<f32> = (0..n * 64).map(|i| i as f32).collect();
    let classes = Dataset {
        x: Tensor::from_vec(x.clone(), &[n, 64]).unwrap(),
        targets: Targets::Classes((0..n).map(|i| i % 10).collect()),
    };
    let values = Dataset {
        x: Tensor::from_vec(x, &[n, 64]).unwrap(),
        targets: Targets::Values(Tensor::full(&[n, 3], 0.5f32)),
    };
    let mut sets = [
        (Batch::new(&classes, 128), classes),
        (Batch::new(&values, 128), values),
    ];
    let mut batchers = [Batcher::new(n, 128, Some(5)), Batcher::new(n, 128, None)];
    let mut sink = 0.0f32;
    let mut before = 0;
    for epoch in 0..4 {
        if epoch == 1 {
            before = allocations();
        }
        for ((out, data), batcher) in sets.iter_mut().zip(&mut batchers) {
            batcher.epoch();
            for range in batcher.batches() {
                let view = batcher.gather(data, range, out);
                sink += view.x.data()[0];
                match view.targets {
                    TargetsView::Classes(c) => sink += c[0] as f32,
                    TargetsView::Values(v) => sink += v.data()[0],
                }
            }
        }
    }
    assert!(sink > 0.0);
    allocations() - before
}

/// The 50k network, 64 → 256 → 128 → 10 with ReLU and He uniform weights, from `Rng::seed(1)`.
fn network_50k(params: &mut ParamStore) -> Sequential {
    let mut rng = Rng::seed(1);
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
        .push(he(256, 128))
        .push(Relu)
        .push(he(128, 10))
}

/// 1,437 rows of 64 features in `[0, 1)` with class labels 0 to 9, from `Rng::seed(2)`: the
/// size of the digits training set, so batch 128 leaves a last batch of 29.
fn synthetic_1437() -> Dataset {
    let n = 1437;
    let mut rng = Rng::seed(2);
    let x: Vec<f32> = (0..n * 64).map(|_| rng.uniform_f32()).collect();
    Dataset {
        x: Tensor::from_vec(x, &[n, 64]).unwrap(),
        targets: Targets::Classes((0..n).map(|_| rng.below(10) as usize).collect()),
    }
}

/// 1,437 rows of 64 features in `[-1, 1)` with class labels 0 to 9, from `Rng::seed(2)`.
/// Centred inputs keep every ReLU neuron of the 50k network active on some row, so a run on
/// them is healthy (the `[0, 1)` set leaves a few neurons dead from the start).
fn synthetic_centred_1437() -> Dataset {
    let n = 1437;
    let mut rng = Rng::seed(2);
    let x: Vec<f32> = (0..n * 64).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    Dataset {
        x: Tensor::from_vec(x, &[n, 64]).unwrap(),
        targets: Targets::Classes((0..n).map(|_| rng.below(10) as usize).collect()),
    }
}

/// The two optimisers the training scenarios run: SGD with momentum and Adam.
fn training_optimizers() -> [Box<dyn Optimizer<f32>>; 2] {
    [
        Box::new(Sgd::new(0.05).with_momentum(0.9)),
        Box::new(Adam::new(1e-3)),
    ]
}

/// Trains the 50k network with `Trainer::step` on the 1,437-row set at batch 128, with SGD
/// and with Adam: one whole warm-up epoch (including the 29-row last batch), then counts
/// the allocations of a second whole epoch.
fn training_step() -> usize {
    let data = synthetic_1437();
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let net = network_50k(&mut params);
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;
        let mut before = 0;
        for epoch in 0..2 {
            if epoch == 1 {
                before = allocations();
            }
            batcher.epoch();
            for range in batcher.batches() {
                let view = batcher.gather(&data, range, &mut batch);
                sink += trainer.step(&net, &mut params, view).unwrap();
            }
        }
        made += allocations() - before;
        assert!(sink.is_finite() && sink > 0.0);
    }
    made
}

/// The manual training loop on the 50k network and the 1,437-row set at batch 128, with SGD
/// and with Adam: a tape on the parameter values, the batch as an input, the forward pass,
/// cross-entropy, `backward` into the gradients, the optimiser step and `zero_grad`. One
/// whole warm-up epoch, then counts the allocations of a second whole epoch.
fn manual_loop() -> usize {
    let data = synthetic_1437();
    let mut made = 0;
    for mut optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let net = network_50k(&mut params);
        let mut graph = Graph::new();
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;
        let mut before = 0;
        for epoch in 0..2 {
            if epoch == 1 {
                before = allocations();
            }
            batcher.epoch();
            for range in batcher.batches() {
                let view = batcher.gather(&data, range, &mut batch);
                let TargetsView::Classes(labels) = view.targets else {
                    unreachable!("the set has class labels")
                };
                let mut tape = graph.tape(&params.values);
                let x = tape.input(view.x);
                let logits = net.forward(&mut tape, x);
                let loss = tape.softmax_cross_entropy(logits, labels, Reduction::Mean);
                sink += tape.value(loss).data()[0];
                tape.backward(loss, &mut params.grads);
                drop(tape);
                optimizer.step(&mut params);
                params.zero_grad();
            }
        }
        made += allocations() - before;
        assert!(sink.is_finite() && sink > 0.0);
    }
    made
}

/// Runs one epoch of `Trainer::step_with_hooks` over `data` at batch 128 and returns the
/// sum of the batch losses.
fn hooked_epoch(
    trainer: &mut Trainer,
    net: &Sequential,
    params: &mut ParamStore,
    data: &Dataset,
    batcher: &mut Batcher,
    batch: &mut Batch,
    hooks: &mut Hooks<'_>,
) -> f32 {
    let mut sum = 0.0;
    batcher.epoch();
    for range in batcher.batches() {
        let view = batcher.gather(data, range, batch);
        sum += trainer.step_with_hooks(net, params, view, hooks).unwrap();
    }
    sum
}

/// Trains the 50k network with `Trainer::step_with_hooks` on the 1,437-row set at batch
/// 128, with SGD and with Adam. With an empty `Hooks`: one warm-up epoch, then counts a
/// second epoch. Then with one forward and one backward hook on the first ReLU layer (whose
/// local derivative is computed into the hooks' own buffer): one warm-up epoch, then counts
/// another. Returns the allocations of both counted epochs.
fn no_hooks_no_cost() -> usize {
    let data = synthetic_1437();
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let net = network_50k(&mut params);
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;

        let mut empty = Hooks::new();
        let mut run = |hooks: &mut Hooks<'_>, sink: &mut f32| {
            *sink += hooked_epoch(
                &mut trainer,
                &net,
                &mut params,
                &data,
                &mut batcher,
                &mut batch,
                hooks,
            );
        };
        run(&mut empty, &mut sink);
        let before = allocations();
        run(&mut empty, &mut sink);
        made += allocations() - before;

        let mut seen = 0.0f32;
        let mut grads = 0.0f32;
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(1, |e| {
                seen += e.output.data()[0] + e.local_derivative.unwrap().data()[0];
            });
            hooks.on_backward(1, |e| grads += e.grad_output.unwrap().data()[0]);
            run(&mut hooks, &mut sink);
            let before = allocations();
            run(&mut hooks, &mut sink);
            made += allocations() - before;
        }
        assert!(sink.is_finite() && sink > 0.0);
        assert!(seen.is_finite() && seen > 0.0 && grads.is_finite());
    }
    made
}

/// Trains the 50k network with neuron masks on both ReLU layers (every third neuron of the
/// first off, every fifth of the second) with `Trainer::step` on the 1,437-row set at batch
/// 128, with SGD and with Adam: one whole warm-up epoch, then counts the allocations of a
/// second whole epoch.
fn masked_training_step() -> usize {
    let data = synthetic_1437();
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let mut net = network_50k(&mut params);
        for (layer, width, every) in [(1, 256, 3), (3, 128, 5)] {
            let mut mask = NeuronMask::all_on(width);
            for neuron in (0..width).step_by(every) {
                mask.set(neuron, false);
            }
            net.set_mask(layer, mask);
        }
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;
        let mut before = 0;
        for epoch in 0..2 {
            if epoch == 1 {
                before = allocations();
            }
            batcher.epoch();
            for range in batcher.batches() {
                let view = batcher.gather(&data, range, &mut batch);
                sink += trainer.step(&net, &mut params, view).unwrap();
            }
        }
        made += allocations() - before;
        assert!(sink.is_finite() && sink > 0.0);
    }
    made
}

/// Looks up parameters of the 50k network by name 1,000 times (each of its six names, one
/// after another) and returns the allocations made.
fn param_lookup() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let names = [
        "0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias",
    ];
    let mut found = 0;
    let before = allocations();
    for i in 0..1000 {
        if net.param(names[i % names.len()]).is_some() {
            found += 1;
        }
        if net.param("9.weight").is_some() || net.param("x").is_some() {
            found += 1;
        }
    }
    let made = allocations() - before;
    assert_eq!(found, 1000);
    made
}

/// Switches neurons of a masked 50k network 1,000 times, on the first and the last `Linear`,
/// and returns the allocations made (the masks exist before counting starts).
fn set_neuron_steady_state() -> usize {
    let mut params = ParamStore::new();
    let mut net = network_50k(&mut params);
    net.set_neuron_mask(0, NeuronMask::all_on(256));
    net.set_neuron_mask(4, NeuronMask::all_on(10));
    let before = allocations();
    for i in 0..1000 {
        let (linear, width) = if i % 2 == 0 { (0, 256) } else { (4, 10) };
        net.set_neuron(linear, i % width, i % 3 == 0);
    }
    let made = allocations() - before;
    assert!(net.neuron_mask(0).unwrap().count_on() < 256);
    made
}

/// Trains the 50k network with neuron masks on both ReLU layers, set with `set_neuron`, with
/// one forward and one backward hook on the first ReLU layer, SGD and Adam, batch 128: one
/// whole warm-up epoch, then counts the allocations of a second whole epoch.
fn masked_hooks_steady_state() -> usize {
    let data = synthetic_1437();
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let mut net = network_50k(&mut params);
        for (linear, width, every) in [(0, 256, 3), (2, 128, 5)] {
            net.set_neuron_mask(linear, NeuronMask::all_on(width));
            for neuron in (0..width).step_by(every) {
                net.set_neuron(linear, neuron, false);
            }
        }
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;
        let mut seen = 0.0f32;
        let mut grads = 0.0f32;
        {
            let mut hooks = Hooks::new();
            hooks.on_forward(1, |e| seen += e.output.data()[1]);
            hooks.on_backward(1, |e| grads += e.grad_output.unwrap().data()[1]);
            let mut before = 0;
            for epoch in 0..2 {
                if epoch == 1 {
                    before = allocations();
                }
                sink += hooked_epoch(
                    &mut trainer,
                    &net,
                    &mut params,
                    &data,
                    &mut batcher,
                    &mut batch,
                    &mut hooks,
                );
            }
            made += allocations() - before;
        }
        assert!(sink.is_finite() && sink > 0.0);
        assert!(seen.is_finite() && grads.is_finite());
    }
    made
}

/// Trains the 50k network with `Trainer::step` on the centred 1,437-row set at batch 128,
/// with SGD and with Adam, with every health check on (non-finite values, gradient norms, dead ReLU
/// neurons) and a sink that counts warnings, calling `Trainer::end_epoch` after each epoch:
/// one whole warm-up epoch, then counts the allocations of a second whole epoch and its end.
/// The run is healthy, so the sink receives nothing.
fn health_steady_state() -> usize {
    let data = synthetic_centred_1437();
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let net = network_50k(&mut params);
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let warnings = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let counter = std::rc::Rc::clone(&warnings);
        trainer.set_health(Health {
            sink: Some(Box::new(move |w: &Warning| {
                eprintln!("health_steady_state: {w}");
                counter.set(counter.get() + 1);
            })),
            ..Health::default()
        });
        let mut batcher = Batcher::new(data.len(), 128, Some(3));
        let mut batch = Batch::new(&data, 128);
        let mut sink = 0.0f32;
        let mut before = 0;
        for epoch in 1..=2 {
            if epoch == 2 {
                before = allocations();
            }
            batcher.epoch();
            for range in batcher.batches() {
                let view = batcher.gather(&data, range, &mut batch);
                sink += trainer.step(&net, &mut params, view).unwrap();
            }
            trainer.end_epoch(epoch).unwrap();
        }
        made += allocations() - before;
        assert!(sink.is_finite() && sink > 0.0);
        assert_eq!(warnings.get(), 0, "the healthy run produced warnings");
    }
    made
}

/// `health_steady_state` under `OnNonFinite::Skip`, on the centred 1,437-row set with a NaN
/// in two rows of different batches (unshuffled, so the batches stay put): every step reads
/// every gradient before the update, and two steps per epoch are skipped. One whole warm-up
/// epoch, then counts the allocations of every step of a second epoch, and of its end,
/// except the one step that sends the epoch's non-finite loss warning (a warning allocates
/// its text); the second skipped step of the epoch, whose warning was already sent, is
/// counted.
fn skip_steady_state() -> usize {
    let mut data = synthetic_centred_1437();
    data.x.data_mut()[200 * 64 + 5] = f32::NAN;
    data.x.data_mut()[900 * 64] = f32::NAN;
    let mut made = 0;
    for optimizer in training_optimizers() {
        let mut params = ParamStore::new();
        let net = network_50k(&mut params);
        let mut trainer = Trainer::new(&net, optimizer, Loss::CrossEntropy, 128);
        let warnings = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let counter = std::rc::Rc::clone(&warnings);
        trainer.set_health(Health {
            on_non_finite: OnNonFinite::Skip,
            sink: Some(Box::new(move |_: &Warning| counter.set(counter.get() + 1))),
            ..Health::default()
        });
        let mut batcher = Batcher::new(data.len(), 128, None);
        let mut batch = Batch::new(&data, 128);
        for epoch in 1..=2 {
            batcher.epoch();
            for range in batcher.batches() {
                let (sent, before) = (warnings.get(), allocations());
                let view = batcher.gather(&data, range, &mut batch);
                trainer.step(&net, &mut params, view).unwrap();
                let step_made = allocations() - before;
                if epoch == 2 && warnings.get() == sent {
                    made += step_made;
                }
            }
            let before = allocations();
            trainer.end_epoch(epoch).unwrap();
            if epoch == 2 {
                made += allocations() - before;
            }
        }
        assert_eq!(trainer.skipped_steps(), 4);
        assert_eq!(warnings.get(), 2, "one non-finite loss warning per epoch");
    }
    made
}

/// Compiles the 50k network with a mask on its second ReLU and runs one input through it
/// once as a warm-up, then 10,000 more inputs. Returns the allocations made after the
/// warm-up.
fn infer_steady_state() -> usize {
    let mut params = ParamStore::new();
    let mut net = network_50k(&mut params);
    let mut mask = NeuronMask::all_on(128);
    mask.set(5, false);
    net.set_mask(3, mask);
    let mut model = InferenceModel::compile(&net, &params).unwrap();
    let data = synthetic_1437();
    let rows = data.x.data();
    let mut sink = model.infer(&rows[..64])[0];
    let before = allocations();
    for i in 0..10_000 {
        let at = (i % 1437) * 64;
        sink += model.infer(&rows[at..at + 64])[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// Compiles the 50k network and runs one batch of 128 rows through `infer_batch` as a
/// warm-up, then 10,000 more batches, alternating 128 and 29 rows (the digits training set's
/// last batch). Returns the allocations made after the warm-up.
fn infer_batch_steady_state() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut model = InferenceModel::compile(&net, &params).unwrap();
    let data = synthetic_1437();
    let rows = data.x.data();
    let batch = |start: usize, n: usize| {
        TensorView::from_slice(&rows[start * 64..(start + n) * 64], &[n, 64]).unwrap()
    };
    let mut sink = model.infer_batch(batch(0, 128)).data()[0];
    let before = allocations();
    for i in 0..10_000 {
        let n = if i % 2 == 0 { 128 } else { 29 };
        let start = (i * 37) % (1437 - 128);
        sink += model.infer_batch(batch(start, n)).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// Compiles the 50k network with a mask on its second ReLU and runs one batch of 128 rows
/// through `infer_batch_observed` as a warm-up, then 2,000 more batches, alternating 128 and
/// 29 rows, summing every observed output. Returns the allocations made after the warm-up.
fn observed_steady_state() -> usize {
    let mut params = ParamStore::new();
    let mut net = network_50k(&mut params);
    net.set_neuron_mask(2, NeuronMask::from_off(128, &[3, 70]));
    let mut model = InferenceModel::compile(&net, &params).unwrap();
    let data = synthetic_1437();
    let rows = data.x.data();
    let batch = |start: usize, n: usize| {
        TensorView::from_slice(&rows[start * 64..(start + n) * 64], &[n, 64]).unwrap()
    };
    let mut sink = 0.0f64;
    let mut layers = 0usize;
    let mut observe = |layer: usize, out: TensorView<'_, f32>| {
        layers += layer;
        sink += f64::from(out.data()[0]);
    };
    model.infer_batch_observed(batch(0, 128), &mut observe);
    let before = allocations();
    for i in 0..2_000 {
        let n = if i % 2 == 0 { 128 } else { 29 };
        let start = (i * 37) % (1437 - 128);
        model.infer_batch_observed(batch(start, n), &mut observe);
    }
    let made = allocations() - before;
    // Layers 1, 3 and 4 are reported on each of the 2,001 calls.
    assert_eq!(layers, 2_001 * 8);
    assert!(sink.is_finite());
    made
}

/// Compiles the 50k network with mask steps on both ReLU layers, then alternates
/// `set_mask` (three prepared masks per layer) with `infer` 1,000 times after one warm-up
/// round. Returns the allocations made after the warm-up.
fn set_mask_steady_state() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut model = InferenceModel::compile_with_masks(&net, &params, &[1, 3]).unwrap();
    let first = [
        NeuronMask::all_on(256),
        NeuronMask::from_off(256, &[0, 9, 255]),
        NeuronMask::from_on(256, &[4]),
    ];
    let second = [
        NeuronMask::from_off(128, &[1]),
        NeuronMask::all_off(128),
        NeuronMask::all_on(128),
    ];
    let data = synthetic_1437();
    let rows = data.x.data();
    model.set_mask(1, &first[0]).unwrap();
    model.set_mask(3, &second[0]).unwrap();
    let mut sink = model.infer(&rows[..64])[0];
    let before = allocations();
    for i in 0..1_000 {
        model.set_mask(1, &first[i % 3]).unwrap();
        model.set_mask(3, &second[(i / 3) % 3]).unwrap();
        let at = (i % 1437) * 64;
        sink += model.infer(&rows[at..at + 64])[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// Traces the 5-layer 50k network (three `Linear` and two `Relu`) into a trace made with
/// `Trace::with_capacity(5)`: one warm-up pass (which starts the tape's buffers), then 100
/// more passes through `forward_traced` with the same trace and graph. Returns the allocations
/// made after the warm-up.
fn trace_with_capacity_allocates_nothing() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let data = synthetic_1437();
    let rows = data.x.data();
    let mut graph = Graph::new();
    let mut trace = Trace::with_capacity(net.len());
    let mut sink = 0usize;
    let mut before = 0;
    for pass in 0..101 {
        if pass == 1 {
            before = allocations();
        }
        let mut tape = graph.tape(&params.values);
        let x = tape.input(TensorView::from_slice(&rows[..128 * 64], &[128, 64]).unwrap());
        net.forward_traced(&mut tape, x, &mut trace);
        sink += trace.len();
    }
    let made = allocations() - before;
    assert_eq!(sink, 101 * 5);
    made
}

/// The `training_step` scenario with Adam, on a trainer that profiled one step of its
/// warm-up epoch (`Trainer::profile_next_step`): the second whole epoch, with no step asking
/// for a profile, must allocate nothing. Returns its allocations.
fn profiling_off_allocates_nothing() -> usize {
    let data = synthetic_1437();
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::CrossEntropy, 128);
    let mut batcher = Batcher::new(data.len(), 128, Some(3));
    let mut batch = Batch::new(&data, 128);
    let mut sink = 0.0f32;
    let mut before = 0;
    for epoch in 0..2 {
        if epoch == 1 {
            before = allocations();
        }
        batcher.epoch();
        for (i, range) in batcher.batches().enumerate() {
            if epoch == 0 && i == 2 {
                trainer.profile_next_step();
            }
            let view = batcher.gather(&data, range, &mut batch);
            sink += trainer.step(&net, &mut params, view).unwrap();
            assert_eq!(trainer.last_profile().is_some(), epoch == 0 && i == 2);
        }
    }
    let made = allocations() - before;
    assert!(sink.is_finite() && sink > 0.0);
    made
}

/// The `training_step` scenario with Adam, profiling one step in each epoch: after the
/// warm-up epoch, whose profiled step makes the profile's buffers, a second whole epoch with
/// one profiled step must allocate nothing, as the profile reuses its buffers. Returns its
/// allocations.
fn profiled_step_reuses_buffers() -> usize {
    let data = synthetic_1437();
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut trainer = Trainer::new(&net, Adam::new(1e-3), Loss::CrossEntropy, 128);
    let mut batcher = Batcher::new(data.len(), 128, Some(3));
    let mut batch = Batch::new(&data, 128);
    let mut sink = 0.0f32;
    let mut total = Duration::ZERO;
    let mut before = 0;
    for epoch in 0..2 {
        if epoch == 1 {
            before = allocations();
        }
        batcher.epoch();
        for (i, range) in batcher.batches().enumerate() {
            if i == 4 {
                trainer.profile_next_step();
            }
            let view = batcher.gather(&data, range, &mut batch);
            sink += trainer.step(&net, &mut params, view).unwrap();
            if let Some(profile) = trainer.last_profile() {
                assert_eq!(profile.forward.len(), net.len());
                assert_eq!(profile.backward.len(), net.len());
                total += profile.total;
            }
        }
    }
    let made = allocations() - before;
    assert!(sink.is_finite() && sink > 0.0 && total > Duration::ZERO);
    made
}

/// The manual training loop of [`manual_loop`] with Adam, through `forward_traced` with a
/// trace whose timing was on for the warm-up epoch and is then switched off: the second
/// epoch, with timing off, must allocate nothing. Returns its allocations.
fn forward_timing_off_allocates_nothing() -> usize {
    let data = synthetic_1437();
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut optimizer = Adam::new(1e-3);
    let mut graph = Graph::new();
    let mut trace = Trace::new();
    trace.set_timing(true);
    let mut batcher = Batcher::new(data.len(), 128, Some(3));
    let mut batch = Batch::new(&data, 128);
    let mut sink = 0.0f32;
    let mut before = 0;
    for epoch in 0..2 {
        if epoch == 1 {
            trace.set_timing(false);
            before = allocations();
        }
        batcher.epoch();
        for range in batcher.batches() {
            let view = batcher.gather(&data, range, &mut batch);
            let TargetsView::Classes(labels) = view.targets else {
                unreachable!("the set has class labels")
            };
            let mut tape = graph.tape(&params.values);
            let x = tape.input(view.x);
            let logits = net.forward_traced(&mut tape, x, &mut trace);
            let loss = tape.softmax_cross_entropy(logits, labels, Reduction::Mean);
            sink += tape.value(loss).data()[0];
            tape.backward(loss, &mut params.grads);
            drop(tape);
            optimizer.step(&mut params);
            params.zero_grad();
        }
    }
    let made = allocations() - before;
    assert!(sink.is_finite() && sink > 0.0);
    made
}

/// The manual training loop with Adam, timed: forward timing on, each layer's output kept
/// as a mark in a reused list, and `backward_timed` into the caller's `times`. Also runs a
/// second backward pass each step into a `ParamGrads::like` set, cleared with its own
/// `zero_grad`, on a graph of its own, as an inspector's probe does. One warm-up epoch, then counts
/// the allocations of a second.
fn timed_step_allocates_nothing() -> usize {
    let data = synthetic_1437();
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut optimizer = Adam::new(1e-3);
    let mut graph = Graph::new();
    let mut probe_graph = Graph::new();
    let mut second = ParamGrads::like(&params.values);
    let mut trace = Trace::new();
    trace.set_timing(true);
    let mut marks: Vec<Var> = Vec::new();
    let mut times = vec![Duration::ZERO; net.len() + 1];
    let mut forward = Duration::ZERO;
    let mut batcher = Batcher::new(data.len(), 128, Some(3));
    let mut batch = Batch::new(&data, 128);
    let mut sink = 0.0f32;
    let mut before = 0;
    for epoch in 0..2 {
        if epoch == 1 {
            before = allocations();
        }
        batcher.epoch();
        for range in batcher.batches() {
            let view = batcher.gather(&data, range, &mut batch);
            let TargetsView::Classes(labels) = view.targets else {
                unreachable!("the set has class labels")
            };
            let mut tape = graph.tape(&params.values);
            let x = tape.input(view.x);
            let logits = net.forward_traced(&mut tape, x, &mut trace);
            let loss = tape.softmax_cross_entropy(logits, labels, Reduction::Mean);
            sink += tape.value(loss).data()[0];
            marks.clear();
            marks.extend((0..trace.len()).map(|i| trace.output(i)));
            forward += (0..trace.len())
                .map(|i| trace.forward_time(i))
                .sum::<Duration>();
            tape.backward_timed(loss, &mut params.grads, &marks, &mut times);
            drop(tape);

            second.zero_grad();
            let mut probe = probe_graph.tape(&params.values);
            let x = probe.input(view.x);
            let logits = net.forward(&mut probe, x);
            let loss = probe.softmax_cross_entropy(logits, labels, Reduction::Mean);
            probe.backward(loss, &mut second);
            drop(probe);

            optimizer.step(&mut params);
            params.zero_grad();
        }
    }
    let made = allocations() - before;
    assert!(sink.is_finite() && sink > 0.0);
    assert!(forward > Duration::ZERO && times.iter().any(|t| *t > Duration::ZERO));
    made
}

/// ReLU written as an activation of your own, so a network using it has no inference form
/// and evaluates on the tape.
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

/// For each row of `data`, whether the two largest outputs of `net` (a chain of `Linear`
/// and `Relu` layers) lie within twice the inference bound of each other, so that two
/// calculation paths may rank them differently. The outputs are computed in `f64`, and the
/// bound is the one the inference tests use: each `Linear` adds
/// `2·(k + 2)·1.19e-7·(Σᵢ |hᵢ|·|wᵢⱼ| + |bⱼ|)` and carries the bound of its input through
/// `Σᵢ boundᵢ·|wᵢⱼ|`; a ReLU carries it unchanged.
fn near_ties(net: &Sequential, params: &ParamStore, data: &Dataset) -> Vec<bool> {
    let features = data.features();
    data.x
        .data()
        .chunks(features)
        .map(|row| {
            let mut h: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            let mut bound = vec![0.0f64; h.len()];
            for i in 0..net.len() {
                let Some(linear) = net.linear(i) else {
                    h.iter_mut().for_each(|v| *v = v.max(0.0));
                    continue;
                };
                let w = params.value(linear.weight()).data();
                let b = params.value(linear.bias()).data();
                let k = h.len();
                let (mut next, mut carried) = (Vec::new(), Vec::new());
                for j in 0..linear.outputs() {
                    let (mut v, mut s, mut c) = (f64::from(b[j]), 0.0f64, 0.0f64);
                    for p in 0..k {
                        let wv = f64::from(w[j * k + p]);
                        v += h[p] * wv;
                        s += h[p].abs() * wv.abs();
                        c += bound[p] * wv.abs();
                    }
                    next.push(v);
                    let own = 2.0 * (k as f64 + 2.0) * 1.19e-7 * (s + f64::from(b[j]).abs());
                    carried.push(own + c);
                }
                h = next;
                bound = carried;
            }
            let mut order: Vec<usize> = (0..h.len()).collect();
            order.sort_by(|&i, &j| h[j].total_cmp(&h[i]));
            let (top, second) = (order[0], order[1]);
            h[top] - h[second] <= 2.0 * (bound[top] + bound[second])
        })
        .collect()
}

/// Evaluates the 50k network on the 1,437-row set at batch 128 (a last batch of 29): with
/// cross-entropy on class labels (`evaluate` and `misclassified`) and with mean squared error
/// on value targets (`evaluate`, with R²), compiled; and the same network with its ReLUs
/// written as an activation of your own, on the tape (`evaluate` and `misclassified` on
/// class labels). One warm-up call of each, then 100 more calls of each. Returns the
/// allocations made after the warm-up.
///
/// The two paths' misclassified rows are compared after the count, leaving out the rows
/// whose two largest outputs are within the calculation bound of each other
/// ([`near_ties`]), where the compiled GEMM and the tape may rank them differently.
fn evaluator_steady_state() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let classes = synthetic_1437();
    let mut rng = Rng::seed(3);
    let y: Vec<f32> = (0..1437 * 10)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let values = Dataset {
        x: classes.x.clone(),
        targets: Targets::Values(Tensor::from_vec(y, &[1437, 10]).unwrap()),
    };
    let mut ce = Evaluator::new(&net, &params, Loss::CrossEntropy, 128).unwrap();
    let mut mse = Evaluator::new(&net, &params, Loss::Mse, 128).unwrap();
    let mut own = Sequential::new();
    for i in 0..net.len() {
        own = match net.linear(i) {
            Some(linear) => own.push(linear.clone()),
            None => own.push(ActivationLayer::new(MyRelu)),
        };
    }
    let mut tape = Evaluator::new(&own, &params, Loss::CrossEntropy, 128).unwrap();
    assert!(format!("{tape:?}").contains("compiled: false"), "{tape:?}");
    let mut wrong = Vec::new();
    let mut tape_wrong = Vec::new();
    let mut sink = 0.0;
    let mut before = 0;
    for call in 0..101 {
        if call == 1 {
            before = allocations();
        }
        let e = ce.evaluate(&classes).unwrap();
        sink += e.loss + e.accuracy.unwrap();
        ce.misclassified(&classes, &mut wrong).unwrap();
        sink += wrong.len() as f64;
        let e = mse.evaluate(&values).unwrap();
        sink += e.loss + e.r2.unwrap();
        let e = tape.evaluate(&classes).unwrap();
        sink += e.loss + e.accuracy.unwrap();
        tape.misclassified(&classes, &mut tape_wrong).unwrap();
        sink += tape_wrong.len() as f64;
    }
    let made = allocations() - before;
    assert!(sink.is_finite() && !wrong.is_empty());
    let ties = near_ties(&net, &params, &classes);
    wrong.retain(|&r| !ties[r]);
    tape_wrong.retain(|&r| !ties[r]);
    assert_eq!(wrong, tape_wrong);
    made
}

/// The 50k network as `Evaluator::with_neuron_masks` on both hidden `Linear`s, compiled,
/// and the same network with its ReLUs written as an activation of your own, on the tape:
/// each switches one neuron of each hidden layer off with `set_neuron_mask` (masks prepared
/// beforehand) and evaluates the 1,437-row set at batch 128 with cross-entropy, 100 times
/// after one warm-up round. Returns the allocations made after the warm-up.
fn evaluator_neuron_mask_steady_state() -> usize {
    let mut params = ParamStore::new();
    let net = network_50k(&mut params);
    let mut own = Sequential::new();
    for i in 0..net.len() {
        own = match net.linear(i) {
            Some(linear) => own.push(linear.clone()),
            None => own.push(ActivationLayer::new(MyRelu)),
        };
    }
    let data = synthetic_1437();
    let first: Vec<NeuronMask> = (0..7)
        .map(|n| NeuronMask::from_off(256, &[n * 31]))
        .collect();
    let second: Vec<NeuronMask> = (0..5)
        .map(|n| NeuronMask::from_off(128, &[n * 17]))
        .collect();
    let mut compiled =
        Evaluator::with_neuron_masks(&net, &params, Loss::CrossEntropy, 128, &[0, 2]).unwrap();
    let mut tape =
        Evaluator::with_neuron_masks(&own, &params, Loss::CrossEntropy, 128, &[0, 2]).unwrap();
    assert!(format!("{tape:?}").contains("compiled: false"), "{tape:?}");
    let mut sink = 0.0;
    let mut before = 0;
    for call in 0..101 {
        if call == 1 {
            before = allocations();
        }
        for evaluator in [&mut compiled, &mut tape] {
            evaluator.set_neuron_mask(0, &first[call % 7]).unwrap();
            evaluator.set_neuron_mask(2, &second[call % 5]).unwrap();
            let e = evaluator.evaluate(&data).unwrap();
            sink += e.loss + e.accuracy.unwrap();
        }
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// Runs every scenario and fails if any allocated after warm-up.
fn main() -> ExitCode {
    let scenarios: [Scenario; 23] = [
        ("optimizers_steady_state", optimizers_steady_state),
        ("optimizers_step_steady_state", optimizers_step_steady_state),
        ("gather_steady_state", gather_steady_state),
        ("training_step", training_step),
        ("manual_loop", manual_loop),
        ("no_hooks_no_cost", no_hooks_no_cost),
        ("masked_training_step", masked_training_step),
        ("param_lookup", param_lookup),
        ("set_neuron_steady_state", set_neuron_steady_state),
        ("masked_hooks_steady_state", masked_hooks_steady_state),
        ("health_steady_state", health_steady_state),
        ("skip_steady_state", skip_steady_state),
        ("infer_steady_state", infer_steady_state),
        ("infer_batch_steady_state", infer_batch_steady_state),
        ("observed_steady_state", observed_steady_state),
        ("set_mask_steady_state", set_mask_steady_state),
        ("evaluator_steady_state", evaluator_steady_state),
        (
            "evaluator_neuron_mask_steady_state",
            evaluator_neuron_mask_steady_state,
        ),
        (
            "trace_with_capacity_allocates_nothing",
            trace_with_capacity_allocates_nothing,
        ),
        (
            "forward_timing_off_allocates_nothing",
            forward_timing_off_allocates_nothing,
        ),
        ("timed_step_allocates_nothing", timed_step_allocates_nothing),
        (
            "profiling_off_allocates_nothing",
            profiling_off_allocates_nothing,
        ),
        ("profiled_step_reuses_buffers", profiled_step_reuses_buffers),
    ];
    let mut failed = false;
    for (name, scenario) in scenarios {
        let made = scenario();
        if made == 0 {
            println!("{name}: ok, 0 allocations");
        } else {
            println!("{name}: FAILED, {made} allocations after warm-up");
            failed = true;
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
