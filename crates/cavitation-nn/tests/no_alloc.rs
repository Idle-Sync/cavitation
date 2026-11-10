//! Counts heap allocations in steady-state steps; any allocation after warm-up fails the run.
//!
//! This target has no test harness: `main` runs each scenario in turn on the main thread, so
//! nothing else allocates while a scenario is being counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use cavitation_core::{Graph, ParamStore, Reduction, Rng, Tensor};
use cavitation_nn::{
    Adam, AdamW, Batch, Batcher, Dataset, GradStats, Hooks, Init, Linear, Loss, NeuronMask,
    Optimizer, Relu, Sequential, Sgd, Targets, TargetsView, Trainer,
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

/// Steps each of the four optimisers 12 times on a store of a matrix and a vector with fresh
/// gradients each step, reusing one `GradStats`. Returns the allocations made during steps
/// 2 to 12 (the first step allocates the state).
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
                sink += trainer.step(&net, &mut params, view);
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
        sum += trainer.step_with_hooks(net, params, view, hooks);
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
                sink += trainer.step(&net, &mut params, view);
            }
        }
        made += allocations() - before;
        assert!(sink.is_finite() && sink > 0.0);
    }
    made
}

/// Runs every scenario and fails if any allocated after warm-up.
fn main() -> ExitCode {
    let scenarios: [Scenario; 7] = [
        ("optimizers_steady_state", optimizers_steady_state),
        ("optimizers_step_steady_state", optimizers_step_steady_state),
        ("gather_steady_state", gather_steady_state),
        ("training_step", training_step),
        ("manual_loop", manual_loop),
        ("no_hooks_no_cost", no_hooks_no_cost),
        ("masked_training_step", masked_training_step),
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
