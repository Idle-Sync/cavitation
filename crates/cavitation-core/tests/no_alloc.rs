//! Counts heap allocations in steady-state steps; any allocation after warm-up fails the run.
//!
//! This target has no test harness: `main` runs each scenario in turn on the main thread, so
//! nothing else allocates while a scenario is being counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use cavitation_core::{Graph, ParamStore, Reduction, Tensor};

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

/// Builds the same small graph 12 times, with a backward pass, a forward-only tape of larger
/// tensors and `zero_grad` each step. Returns the allocations made during steps 3 to 12.
fn tape_steady_state() -> usize {
    let mut store = ParamStore::new();
    let p1 = store.add(Tensor::from_vec(vec![0.5f32], &[1]).unwrap());
    let p2 = store.add(Tensor::from_vec(vec![-1.5f32], &[1]).unwrap());
    let big = store.add(Tensor::full(&[3, 4], 2.0f32));
    let x = Tensor::from_vec(vec![3.0f32], &[1]).unwrap();
    let wide = Tensor::full(&[3, 4], 1.0f32);
    let row = store.add(Tensor::full(&[4], 0.5f32));
    let col = store.add(Tensor::full(&[3, 1], 0.25f32));
    let cell = Tensor::full(&[1, 1], 2.0f32);
    let rowvec = Tensor::full(&[1, 4], 0.5f32);
    let mw = store.add(Tensor::full(&[4, 3], 0.25f32));
    let mq = store.add(Tensor::full(&[3, 2], 0.5f32));
    let mv = store.add(Tensor::full(&[1, 2], 0.75f32));
    let mut graph = Graph::new();
    let mut sink = 0.0f32;
    let mut before = 0;
    for step in 1..=12 {
        if step == 3 {
            before = allocations();
        }
        store.zero_grad();
        {
            let mut tape = graph.tape(&store.values);
            let xi = tape.input(x.view());
            let (a, b) = (tape.param(p1), tape.param(p2));
            let h = tape.add(xi, a);
            let h2 = tape.add(h, b);
            let loss = tape.add(h2, h2);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0] + tape.grad(h).map_or(0.0, |g| g.data()[0]);
        }
        {
            let mut tape = graph.tape(&store.values);
            let w = tape.input(wide.view());
            let p = tape.param(big);
            let s = tape.add(w, p);
            let t = tape.add(s, s);
            sink += tape.value(t).data()[11];
        }
        {
            // Broadcast add (the bias add), outer-product sub, broadcast mul and scale on
            // larger tensors, forward only.
            let mut tape = graph.tape(&store.values);
            let w = tape.input(wide.view());
            let (r, c) = (tape.param(row), tape.param(col));
            let h = tape.add(w, r);
            let d = tape.sub(h, c);
            let m = tape.mul(d, r);
            let out = tape.scale(m, 0.5);
            sink += tape.value(out).data()[11];
        }
        {
            // The same operations with a backward pass, on shapes that broadcast to one
            // element, so each input's gradient is reduced through a scratch buffer.
            let mut tape = graph.tape(&store.values);
            let xi = tape.input(cell.view());
            let (a, b) = (tape.param(p1), tape.param(p2));
            let h = tape.add(xi, a);
            let d = tape.sub(h, b);
            let m = tape.mul(d, a);
            let loss = tape.scale(m, 0.5);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
        }
        {
            // matmul and matmul_nt, forward and backward, ending in a 1 x 1 result:
            // [1, 4] . [4, 3] . [3, 2] -> [1, 2], then . [1, 2]^T -> [1, 1]. All three
            // weights take gradients; the input takes none.
            let mut tape = graph.tape(&store.values);
            let xi = tape.input(rowvec.view());
            let (w, v) = (tape.param(mw), tape.param(mv));
            let h = tape.matmul(xi, w);
            let q = tape.param(mq);
            let g = tape.matmul(h, q);
            let loss = tape.matmul_nt(g, v);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0] + store.grad(mw).data()[0];
        }
        sink += store.grad(p1).data()[0] + store.grad(big).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// A `[8, 16] + [16]` bias add, ReLU and sum, forward and backward, 12 times: the bias
/// gradient is reduced over the rows through a scratch buffer. Returns the allocations made
/// during steps 3 to 12.
fn bias_relu_sum_steady_state() -> usize {
    let mut store = ParamStore::new();
    let bias = store.add(Tensor::full(&[16], 0.1f32));
    let weight = store.add(Tensor::full(&[8, 16], -0.3f32));
    let x = Tensor::from_vec(
        (0..128).map(|i| (i as f32 * 0.37).sin()).collect(),
        &[8, 16],
    )
    .unwrap();
    let mut graph = Graph::new();
    let mut sink = 0.0f32;
    let mut before = 0;
    for step in 1..=12 {
        if step == 3 {
            before = allocations();
        }
        store.zero_grad();
        {
            let mut tape = graph.tape(&store.values);
            let xi = tape.input(x.view());
            let (b, w) = (tape.param(bias), tape.param(weight));
            let h = tape.add(xi, b);
            let h = tape.add(h, w);
            let r = tape.relu(h);
            let t = tape.tanh(r);
            let loss = tape.sum(t);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
        }
        sink += store.grad(bias).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// Softmax cross-entropy and MSE, forward and backward, 12 times: a batch of 8 samples with
/// 10 classes through a linear layer, the labels passed per step as a slice. Returns the
/// allocations made during steps 3 to 12.
fn cross_entropy_steady_state() -> usize {
    let mut store = ParamStore::new();
    let weight = store.add(Tensor::full(&[4, 10], 0.1f32));
    let bias = store.add(Tensor::full(&[10], 0.0f32));
    let x = Tensor::from_vec((0..32).map(|i| (i as f32 * 0.7).cos()).collect(), &[8, 4]).unwrap();
    let target = Tensor::full(&[8, 10], 0.5f32);
    let labels = [3usize, 0, 9, 4, 4, 1, 7, 2];
    let mut graph = Graph::new();
    let mut sink = 0.0f32;
    let mut before = 0;
    for step in 1..=12 {
        if step == 3 {
            before = allocations();
        }
        store.zero_grad();
        {
            let mut tape = graph.tape(&store.values);
            let xi = tape.input(x.view());
            let (w, b) = (tape.param(weight), tape.param(bias));
            let h = tape.matmul(xi, w);
            let logits = tape.add(h, b);
            let ce = tape.softmax_cross_entropy(logits, &labels, Reduction::Mean);
            let lsm = tape.log_softmax(logits);
            let sm = tape.softmax(lsm);
            let t = tape.input(target.view());
            let err = tape.mse(sm, t, Reduction::BatchMean);
            let loss = tape.add(ce, err);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
        }
        sink += store.grad(weight).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// A named scenario: runs, and returns the allocations it made after warm-up.
type Scenario = (&'static str, fn() -> usize);

/// Runs every scenario in order and fails if any of them allocated after warm-up.
fn main() -> ExitCode {
    let scenarios: [Scenario; 3] = [
        ("tape_steady_state", tape_steady_state),
        ("bias_relu_sum_steady_state", bias_relu_sum_steady_state),
        ("cross_entropy_steady_state", cross_entropy_steady_state),
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
