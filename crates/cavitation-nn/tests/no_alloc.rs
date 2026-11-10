//! Counts heap allocations in steady-state steps; any allocation after warm-up fails the run.
//!
//! This target has no test harness: `main` runs each scenario in turn on the main thread, so
//! nothing else allocates while a scenario is being counted.

use std::alloc::{GlobalAlloc, Layout, System};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use cavitation_core::{ParamStore, Tensor};
use cavitation_nn::{Adam, AdamW, GradStats, Optimizer, Sgd};

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

/// Runs every scenario and fails if any allocated after warm-up.
fn main() -> ExitCode {
    let scenarios: [Scenario; 1] = [("optimizers_steady_state", optimizers_steady_state)];
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
