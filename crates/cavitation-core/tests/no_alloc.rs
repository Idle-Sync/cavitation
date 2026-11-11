//! Counts heap allocations in steady-state steps; any allocation after warm-up fails the run.
//!
//! This target has no test harness: `main` runs each scenario in turn on the main thread, so
//! nothing else allocates while a scenario is being counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use cavitation_core::kernels::gemm::gemm_f32_with;
use cavitation_core::kernels::pool::{self, Task};
use cavitation_core::kernels::{MatMut, MatRef, level};

use std::sync::Arc;

use cavitation_core::{
    CustomOp, Graph, ParamStore, Real, Reduction, Shape, Tensor, TensorView, TensorViewMut,
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

/// Builds the same small graphs 12 times, with backward passes (`keep_where` among them), a
/// forward-only tape of larger tensors and `zero_grad` each step. Returns the allocations made during steps 3 to 12.
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
    let keep = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 1.0], &[4]).unwrap();
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
        {
            // keep_where with a broadcast keep row, forward and backward, as a neuron mask
            // records it: [3, 4] parameter, [4] keep, then a sum to one value.
            let mut tape = graph.tape(&store.values);
            let p = tape.param(big);
            let k = tape.input(keep.view());
            let masked = tape.keep_where(p, k);
            let loss = tape.sum(masked);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
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

/// Binary cross-entropy with logits and NLL, forward and backward, 12 times: a batch of 8
/// samples with 10 classes through a linear layer. Returns the allocations made during steps
/// 3 to 12.
fn bce_and_nll_steady_state() -> usize {
    let mut store = ParamStore::new();
    let weight = store.add(Tensor::full(&[4, 10], 0.1f32));
    let x = Tensor::from_vec((0..32).map(|i| (i as f32 * 0.7).cos()).collect(), &[8, 4]).unwrap();
    let target = Tensor::full(&[8, 10], 0.25f32);
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
            let w = tape.param(weight);
            let logits = tape.matmul(xi, w);
            let t = tape.input(target.view());
            let bce = tape.binary_cross_entropy_with_logits(logits, t, Reduction::Mean);
            let lsm = tape.log_softmax(logits);
            let nll = tape.nll(lsm, &labels, Reduction::Mean);
            let loss = tape.add(bce, nll);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
        }
        sink += store.grad(weight).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// A user-defined elementwise multiply, `a · b`, for the custom-operation scenario.
struct CustomMul;

impl<T: Real> CustomOp<T> for CustomMul {
    fn name(&self) -> &str {
        "custom_mul"
    }

    fn output_shape(&self, inputs: &[Shape]) -> Shape {
        inputs[0]
    }

    fn forward(&self, inputs: &[TensorView<T>], mut output: TensorViewMut<T>) {
        let (a, b) = (inputs[0].data(), inputs[1].data());
        for (i, o) in output.data_mut().iter_mut().enumerate() {
            *o = a[i] * b[i];
        }
    }

    fn backward(
        &self,
        inputs: &[TensorView<T>],
        _output: TensorView<T>,
        grad_output: TensorView<T>,
        grad_inputs: &mut [TensorViewMut<T>],
    ) {
        let (a, b, g) = (inputs[0].data(), inputs[1].data(), grad_output.data());
        for i in 0..g.len() {
            grad_inputs[0].data_mut()[i] = b[i] * g[i];
            grad_inputs[1].data_mut()[i] = a[i] * g[i];
        }
    }
}

/// A custom two-input multiply, forward and backward, 12 times: once with a parameter and
/// an input, once with the same parameter in both slots. Returns the allocations made during
/// steps 3 to 12.
fn custom_op_steady_state() -> usize {
    let mut store = ParamStore::new();
    let weight = store.add(Tensor::full(&[4, 3], 0.5f32));
    let x = Tensor::from_vec((0..12).map(|i| (i as f32 * 0.3).sin()).collect(), &[4, 3]).unwrap();
    let op: Arc<dyn CustomOp<f32> + Send + Sync> = Arc::new(CustomMul);
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
            let w = tape.param(weight);
            let h = tape.custom(&op, &[xi, w]);
            let sq = tape.custom(&op, &[h, h]);
            let loss = tape.sum(sq);
            tape.backward(loss, &mut store.grads);
            sink += tape.value(loss).data()[0];
        }
        sink += store.grad(weight).data()[0];
    }
    let made = allocations() - before;
    assert!(sink.is_finite());
    made
}

/// The thread pool on 4 threads: 100 warm-up runs (which start the workers), then 1,000
/// runs of 64 tasks each. Returns the allocations made during those 1,000 runs, on any thread.
fn pool_steady_state() -> usize {
    let sum = AtomicUsize::new(0);
    let job = |t: Task| {
        sum.fetch_add(t.index + t.worker, Ordering::Relaxed);
    };
    for _ in 0..100 {
        pool::run(4, 64, &job);
    }
    let before = allocations();
    for _ in 0..1_000 {
        pool::run(4, 64, &job);
    }
    let made = allocations() - before;
    assert!(sum.load(Ordering::Relaxed) > 0);
    made
}

/// The pool's slow path: the caller sleeps on the pool's condition variable for the first
/// time in the counted rounds. Runs first, so no earlier scenario has made the caller sleep.
/// Warm-up runs 100 instant jobs. In each counted round the caller's task waits until the
/// three worker tasks have started (so the caller cannot take them itself), and each worker
/// task then sleeps 2 ms, longer than the caller spins, so the caller must sleep on the
/// condition variable. On Apple targets that variable and its lock allocate on first use
/// unless the pool has used them before. Returns the allocations made during the counted
/// rounds, on any thread.
fn pool_wait_steady_state() -> usize {
    let slow = AtomicBool::new(false);
    let started = AtomicUsize::new(0);
    let job = |t: Task| {
        if !slow.load(Ordering::Relaxed) {
            return;
        }
        if t.worker == 0 {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while started.load(Ordering::Acquire) < 3 && std::time::Instant::now() < deadline {
                std::hint::spin_loop();
            }
        } else {
            started.fetch_add(1, Ordering::Release);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    };
    for _ in 0..100 {
        pool::run(4, 4, &job);
    }
    slow.store(true, Ordering::Relaxed);
    let before = allocations();
    for _ in 0..20 {
        started.store(0, Ordering::Relaxed);
        pool::run(4, 4, &job);
    }
    allocations() - before
}

/// The f32 GEMM on 4 threads at the current kernel level, alternating a large forward-pass
/// product (128 x 1024 x 784, B transposed) and a small one (32 x 10 x 256): 3 warm-up rounds
/// (which size the packing buffers), then 10 counted rounds. Returns the allocations made
/// during the counted rounds, on any thread.
fn gemm_steady_state() -> usize {
    let a1 = vec![0.25f32; 128 * 784];
    let w1 = vec![0.5f32; 1024 * 784];
    let mut c1 = vec![0.0f32; 128 * 1024];
    let a2 = vec![0.5f32; 32 * 256];
    let b2 = vec![0.25f32; 256 * 10];
    let mut c2 = vec![0.0f32; 32 * 10];
    let level = level();
    let mut round = |beta: f32| {
        gemm_f32_with(
            level,
            4,
            1.0,
            MatRef::new(&a1, 128, 784, 784, 1).unwrap(),
            MatRef::new(&w1, 1024, 784, 784, 1).unwrap().transpose(),
            beta,
            MatMut::new(&mut c1, 128, 1024, 1024, 1).unwrap(),
        );
        gemm_f32_with(
            level,
            4,
            1.0,
            MatRef::new(&a2, 32, 256, 256, 1).unwrap(),
            MatRef::new(&b2, 256, 10, 10, 1).unwrap(),
            beta,
            MatMut::new(&mut c2, 32, 10, 10, 1).unwrap(),
        );
    };
    for _ in 0..3 {
        round(0.0);
    }
    let before = allocations();
    for i in 0..10 {
        round(if i % 2 == 0 { 0.0 } else { 1.0 });
    }
    let made = allocations() - before;
    assert!(c1[0].is_finite() && c2[0].is_finite());
    made
}

/// A named scenario: runs, and returns the allocations it made after warm-up.
type Scenario = (&'static str, fn() -> usize);

/// Runs every scenario in order and fails if any of them allocated after warm-up.
fn main() -> ExitCode {
    let scenarios: [Scenario; 8] = [
        ("pool_wait_steady_state", pool_wait_steady_state),
        ("tape_steady_state", tape_steady_state),
        ("bias_relu_sum_steady_state", bias_relu_sum_steady_state),
        ("cross_entropy_steady_state", cross_entropy_steady_state),
        ("bce_and_nll_steady_state", bce_and_nll_steady_state),
        ("custom_op_steady_state", custom_op_steady_state),
        ("pool_steady_state", pool_steady_state),
        ("gemm_steady_state", gemm_steady_state),
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
