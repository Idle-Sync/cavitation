//! A persistent thread pool that runs a numbered set of tasks and returns when all are done.
//!
//! The workers are started on first use and live for the rest of the process. After a job a
//! worker spins for about 200 microseconds, so the next job in a burst (the matrix products
//! of one training step) starts without a wake-up, then parks. Running a job allocates
//! nothing once the pool has as many workers as the job asks for.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// The most threads, the caller included, that one call uses.
const MAX_THREADS: usize = 256;

/// How long a worker, or a caller waiting for workers, spins before it sleeps.
const SPIN: Duration = Duration::from_micros(200);

/// One task of a [`run`] call, as passed to the job.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::pool::{run, Task};
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// let total = AtomicUsize::new(0);
/// run(2, 4, &|t: Task| {
///     assert!(t.worker < 2);
///     total.fetch_add(t.index, Ordering::Relaxed);
/// });
/// assert_eq!(total.into_inner(), 0 + 1 + 2 + 3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Task {
    /// The task's number, in `0..tasks`. Each number is run exactly once per call.
    pub index: usize,
    /// The thread running the task, in `0..threads`; 0 is the calling thread. Within one
    /// call, two tasks running at the same time never share a worker number; tasks of
    /// different calls (concurrent or nested) can share one. Per-worker scratch space must
    /// therefore belong to the call (or be thread-local), never be shared between calls.
    /// Which worker runs which task varies from call to call.
    pub worker: usize,
}

/// The job of the call in progress, lifetime erased.
#[derive(Clone, Copy)]
struct Job {
    /// The caller's job. `None` only before the first call.
    func: Option<NonNull<dyn Fn(Task) + Sync + 'static>>,
    /// The number of tasks.
    tasks: usize,
}

/// A pool thread and what it needs to be woken.
struct Worker {
    /// The worker's number in [`Task::worker`]; at least 1.
    id: usize,
    /// The number of the last call this worker was given; a new value means a new job.
    posted: AtomicUsize,
    /// Whether the worker is parked, or about to park, and needs an unpark to see a job.
    parked: AtomicBool,
    /// The worker's thread, for unparking; set once, right after it is spawned.
    thread: OnceLock<Thread>,
}

/// The pool's shared state. There is one, in [`POOL`].
struct Pool {
    /// Held by the one caller whose job the workers are running; see [`run`].
    busy: AtomicBool,
    /// The current job. Written only by the holder of `busy` while no worker is between
    /// seeing its `posted` change and decrementing `remaining`; read only by such workers.
    job: UnsafeCell<Job>,
    /// The next task number to hand out.
    next: AtomicUsize,
    /// The number of the last call, counted from 1.
    call: AtomicUsize,
    /// The workers given the current job that have not yet finished with it.
    remaining: AtomicUsize,
    /// Whether the caller is asleep on `done` waiting for `remaining` to reach 0.
    caller_waiting: AtomicBool,
    /// The lock `done` waits with.
    done_lock: Mutex<()>,
    /// Signalled by the last worker to finish a job while the caller waits.
    done: Condvar,
    /// The first panic payload of the current job.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// The workers started so far; worker `i + 1` is at index `i`. Grows only.
    workers: Mutex<Vec<&'static Worker>>,
}

// SAFETY: every field but `job` is `Sync`. `job` is written only by the thread holding
// `busy`, and only when no worker is reading it: the workers that read it are exactly those
// whose `posted` the holder changed, they read it after seeing that change (acquire) and
// before decrementing `remaining` (release), and the holder writes it again only after
// seeing `remaining` reach 0 (acquire) and reacquiring `busy`. The `NonNull` it holds points
// to a `dyn Fn + Sync`, which may be called from any thread.
unsafe impl Sync for Pool {}

/// The process-wide pool.
static POOL: Pool = Pool {
    busy: AtomicBool::new(false),
    job: UnsafeCell::new(Job {
        func: None,
        tasks: 0,
    }),
    next: AtomicUsize::new(0),
    call: AtomicUsize::new(0),
    remaining: AtomicUsize::new(0),
    caller_waiting: AtomicBool::new(false),
    done_lock: Mutex::new(()),
    done: Condvar::new(),
    panic: Mutex::new(None),
    workers: Mutex::new(Vec::new()),
};

thread_local! {
    /// Whether this thread is one of the pool's workers.
    static IS_WORKER: Cell<bool> = const { Cell::new(false) };
}

/// Locks `m`, ignoring poisoning: nothing panics while these locks are held.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs `job` once for every task number in `0..tasks`, on at most `threads` threads
/// including the calling one, and returns when every task has finished.
///
/// The pool's workers are started the first time a call needs them and then reused; a call
/// with more threads than there are workers starts the missing ones. A call made while
/// another thread's call is using the pool, or from inside a task (a nested call), runs all
/// of its tasks on the calling thread as worker 0. Calls allocate nothing once the pool has
/// enough workers.
///
/// Tasks are handed out to threads in no fixed order. For results that are the same for any
/// thread count and any schedule, each task's output must depend only on [`Task::index`]
/// (for example, each task writes its own block of the output); use [`Task::worker`] only to
/// choose per-thread scratch space owned by this call. Worker numbers are unique only within
/// one call: a concurrent or nested call also uses worker 0, so scratch indexed by worker
/// number must not be shared between calls.
///
/// At most 256 threads take part, whatever `threads` asks for; results are unchanged, since
/// each task's output depends only on its index.
///
/// # Panics
///
/// Panics if `threads` is 0. If tasks panic, the remaining tasks still run, and then the
/// first panic is resumed on the calling thread; the pool stays usable.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::pool::{run, Task};
/// use std::sync::atomic::{AtomicU32, Ordering};
/// let out: Vec<AtomicU32> = (0..8).map(|_| AtomicU32::new(0)).collect();
/// run(4, 8, &|t: Task| out[t.index].store(t.index as u32 * 10, Ordering::Relaxed));
/// assert_eq!(out[7].load(Ordering::Relaxed), 70);
/// ```
pub fn run(threads: usize, tasks: usize, job: &(dyn Fn(Task) + Sync)) {
    assert!(threads >= 1, "pool::run: threads must be at least 1, got 0");
    let wanted = threads.min(tasks).min(MAX_THREADS);
    if wanted <= 1 || IS_WORKER.get() {
        return run_inline(tasks, job);
    }
    if POOL
        .busy
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return run_inline(tasks, job);
    }
    run_on_pool(wanted, tasks, job);
}

/// Runs every task on the calling thread as worker 0, then resumes the first panic, if any.
fn run_inline(tasks: usize, job: &(dyn Fn(Task) + Sync)) {
    let mut first_panic = None;
    let next = AtomicUsize::new(0);
    while next.load(Ordering::Relaxed) < tasks {
        let result = catch_unwind(AssertUnwindSafe(|| {
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= tasks {
                    break;
                }
                job(Task { index, worker: 0 });
            }
        }));
        if let Err(payload) = result {
            first_panic.get_or_insert(payload);
        }
    }
    if let Some(payload) = first_panic {
        resume_unwind(payload);
    }
}

/// Runs the job on the caller and `threads - 1` workers. The caller holds `busy`; this
/// releases it before returning.
fn run_on_pool(threads: usize, tasks: usize, job: &(dyn Fn(Task) + Sync)) {
    let workers = lock(&POOL.workers);
    let started = grow(workers, threads - 1);
    if started.is_empty() {
        // Unlock the worker list before running the tasks: a nested call from one of them
        // would otherwise lock it again on this thread and deadlock.
        drop(started);
        POOL.busy.store(false, Ordering::Release);
        return run_inline(tasks, job);
    }
    // SAFETY: only the lifetime changes; the pointer stays valid for as long as any worker
    // can use it. Workers call it only between seeing their `posted` change below and
    // decrementing `remaining`, and this function does not return (or let the caller's
    // borrow of `job` end) until `remaining` is 0. Nothing between the publish and that wait
    // can unwind: task panics are caught, and the locks ignore poisoning.
    let func = unsafe {
        std::mem::transmute::<
            NonNull<dyn Fn(Task) + Sync + '_>,
            NonNull<dyn Fn(Task) + Sync + 'static>,
        >(NonNull::from(job))
    };
    // SAFETY: we hold `busy` and the last job's workers have all decremented `remaining`
    // (the previous holder waited for that), so no other thread is reading `job`.
    unsafe {
        *POOL.job.get() = Job {
            func: Some(func),
            tasks,
        };
    }
    POOL.next.store(0, Ordering::Relaxed);
    POOL.remaining.store(started.len(), Ordering::Relaxed);
    let call = POOL.call.load(Ordering::Relaxed).wrapping_add(1);
    POOL.call.store(call, Ordering::Relaxed);
    for w in &started {
        // SeqCst pairs with the worker's store to `parked` and load of `posted`: either the
        // worker sees this call, or we see that it is parked and unpark it.
        w.posted.store(call, Ordering::SeqCst);
        if w.parked.load(Ordering::SeqCst)
            && let Some(t) = w.thread.get()
        {
            t.unpark();
        }
    }
    drop(started);
    work(tasks, job, 0);
    wait_for_workers();
    let payload = lock(&POOL.panic).take();
    POOL.busy.store(false, Ordering::Release);
    if let Some(payload) = payload {
        resume_unwind(payload);
    }
}

/// Starts workers until there are at least `wanted`, and returns the guard holding the list
/// with the first `wanted` (fewer if the system refuses to start more threads).
fn grow(mut workers: MutexGuard<'static, Vec<&'static Worker>>, wanted: usize) -> WorkerList {
    while workers.len() < wanted {
        // The call number before the next call: the worker's first job is the one after it.
        let last_call = POOL.call.load(Ordering::Relaxed);
        let worker: &'static Worker = Box::leak(Box::new(Worker {
            id: workers.len() + 1,
            posted: AtomicUsize::new(last_call),
            parked: AtomicBool::new(false),
            thread: OnceLock::new(),
        }));
        match spawn_worker(worker, last_call) {
            Ok(handle) => {
                let _ = worker.thread.set(handle.thread().clone());
                workers.push(worker);
            }
            Err(_) => break,
        }
    }
    let len = workers.len().min(wanted);
    WorkerList {
        guard: workers,
        len,
    }
}

/// Makes every later worker spawn fail, so tests can reach the path where the pool cannot
/// start any thread. Set only from a test running alone in a child process.
#[cfg(test)]
static FAIL_SPAWN: AtomicBool = AtomicBool::new(false);

/// Starts the thread for `worker`, created after call number `last_call`.
fn spawn_worker(
    worker: &'static Worker,
    last_call: usize,
) -> std::io::Result<thread::JoinHandle<()>> {
    #[cfg(test)]
    if FAIL_SPAWN.load(Ordering::Relaxed) {
        return Err(std::io::Error::other("worker spawn disabled for a test"));
    }
    thread::Builder::new()
        .name(format!("cavitation-pool-{}", worker.id))
        .spawn(move || worker_main(worker, last_call))
}

/// The first `len` workers, with the list kept locked.
struct WorkerList {
    /// The locked list of every worker.
    guard: MutexGuard<'static, Vec<&'static Worker>>,
    /// How many of them take part in the current call.
    len: usize,
}

impl WorkerList {
    /// Whether no worker takes part.
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many workers take part.
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a> IntoIterator for &'a WorkerList {
    type Item = &'a &'static Worker;
    type IntoIter = std::slice::Iter<'a, &'static Worker>;

    /// The workers taking part.
    fn into_iter(self) -> Self::IntoIter {
        self.guard[..self.len].iter()
    }
}

/// Claims and runs tasks until none are left, as `worker`, recording the first panic in
/// [`Pool::panic`] and carrying on after it.
fn work(tasks: usize, job: &(dyn Fn(Task) + Sync), worker: usize) {
    while POOL.next.load(Ordering::Relaxed) < tasks {
        let result = catch_unwind(AssertUnwindSafe(|| {
            loop {
                let index = POOL.next.fetch_add(1, Ordering::Relaxed);
                if index >= tasks {
                    break;
                }
                job(Task { index, worker });
            }
        }));
        if let Err(payload) = result {
            let mut first = lock(&POOL.panic);
            if first.is_none() {
                *first = Some(payload);
            } else {
                drop(first);
                // A later payload is leaked, not dropped: its destructor could panic, and an
                // unwind out of here could let the caller return while workers still use the
                // job, or end a worker before it reports done. Leaks only on this path.
                std::mem::forget(payload);
            }
        }
    }
}

/// Waits until every worker given the current job has finished with it: spins for
/// [`SPIN`], then sleeps on [`Pool::done`].
fn wait_for_workers() {
    let start = Instant::now();
    let mut spins = 0u32;
    while POOL.remaining.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
        spins = spins.wrapping_add(1);
        if spins.is_multiple_of(64) {
            // Give the core away now and then. When other processes keep a core busy, a
            // pool thread that has been preempted otherwise waits for a pure spinner to
            // reach its time limit: about 210 us per round trip, against 0.5 to 1 us with
            // these yields under light load and 11 to 24 us under heavy load (empty 4-thread
            // job on a 4-core machine, measured with `pool_round_trip_time`).
            thread::yield_now();
        }
        if spins.is_multiple_of(64) && start.elapsed() >= SPIN {
            let mut guard = lock(&POOL.done_lock);
            // SeqCst pairs with the worker's decrement and load of `caller_waiting`: either
            // we see `remaining` at 0, or the last worker sees us waiting and signals.
            POOL.caller_waiting.store(true, Ordering::SeqCst);
            while POOL.remaining.load(Ordering::SeqCst) != 0 {
                guard = POOL
                    .done
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            POOL.caller_waiting.store(false, Ordering::SeqCst);
            return;
        }
    }
}

/// A worker thread's loop: wait for a job, help run its tasks, report that it is done.
/// `seen` is the call number the worker was created after; it may start running after the
/// next call has already been posted to it.
fn worker_main(me: &'static Worker, mut seen: usize) {
    IS_WORKER.set(true);
    loop {
        seen = wait_for_job(me, seen);
        // SAFETY: we saw `posted` change to this call (acquire), which the caller stored
        // after writing `job`, and the caller does not write `job` again until we decrement
        // `remaining` below. The function pointer stays valid until then for the same
        // reason: the caller is blocked in `wait_for_workers`.
        let job = unsafe { *POOL.job.get() };
        if let Some(func) = job.func {
            // SAFETY: as above, the caller keeps the job borrowed until we decrement
            // `remaining`, and the job is `Sync`, so calling it from this thread is sound.
            let func = unsafe { func.as_ref() };
            work(job.tasks, func, me.id);
        }
        // From here on we touch only `POOL` (static), never `job`.
        if POOL.remaining.fetch_sub(1, Ordering::SeqCst) == 1
            && POOL.caller_waiting.load(Ordering::SeqCst)
        {
            let _guard = lock(&POOL.done_lock);
            POOL.done.notify_one();
        }
    }
}

/// Waits until `me.posted` differs from `seen` and returns the new value: spins for
/// [`SPIN`], then parks.
fn wait_for_job(me: &Worker, seen: usize) -> usize {
    let start = Instant::now();
    let mut spins = 0u32;
    loop {
        let posted = me.posted.load(Ordering::Acquire);
        if posted != seen {
            return posted;
        }
        std::hint::spin_loop();
        spins = spins.wrapping_add(1);
        if spins.is_multiple_of(64) {
            // Give the core away now and then. When other processes keep a core busy, a
            // pool thread that has been preempted otherwise waits for a pure spinner to
            // reach its time limit: about 210 us per round trip, against 0.5 to 1 us with
            // these yields under light load and 11 to 24 us under heavy load (empty 4-thread
            // job on a 4-core machine, measured with `pool_round_trip_time`).
            thread::yield_now();
        }
        if spins.is_multiple_of(64) && start.elapsed() >= SPIN {
            break;
        }
    }
    loop {
        // SeqCst pairs with the caller's store to `posted` and load of `parked`.
        me.parked.store(true, Ordering::SeqCst);
        let posted = me.posted.load(Ordering::SeqCst);
        if posted != seen {
            me.parked.store(false, Ordering::Relaxed);
            return posted;
        }
        thread::park();
        me.parked.store(false, Ordering::Relaxed);
    }
}

/// Tests for the thread pool.
#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    /// Held by every test that needs the pool to itself, so a call is not sent to the inline
    /// path just because another test's call holds the pool.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Takes [`SERIAL`], ignoring poisoning from a failed test.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Runs `tasks` tasks on `threads` threads and returns how many times each index ran.
    fn counts(threads: usize, tasks: usize) -> Vec<usize> {
        let seen: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
        run(threads, tasks, &|t: Task| {
            assert!(
                t.worker < threads,
                "worker {} with {threads} threads",
                t.worker
            );
            seen[t.index].fetch_add(1, Ordering::Relaxed);
        });
        seen.into_iter().map(AtomicUsize::into_inner).collect()
    }

    /// 10,000 tasks on 4 threads: every index runs exactly once.
    #[test]
    fn pool_runs_every_task_once() {
        let _serial = serial();
        assert!(counts(4, 10_000).iter().all(|&c| c == 1));
        assert!(counts(1, 100).iter().all(|&c| c == 1));
        assert!(counts(4, 0).is_empty());
    }

    /// 16 threads work whatever the number of cores.
    #[test]
    fn pool_more_threads_than_cores() {
        let _serial = serial();
        for _ in 0..20 {
            assert!(counts(16, 1_000).iter().all(|&c| c == 1));
        }
    }

    /// A panic in one task reaches the caller after every other task has run, and the pool
    /// still works afterwards.
    #[test]
    fn pool_propagates_panic() {
        let _serial = serial();
        let seen: Vec<AtomicUsize> = (0..100).map(|_| AtomicUsize::new(0)).collect();
        let result = catch_unwind(AssertUnwindSafe(|| {
            run(4, 100, &|t: Task| {
                seen[t.index].fetch_add(1, Ordering::Relaxed);
                if t.index == 37 {
                    panic!("task 37 failed");
                }
            });
        }));
        let payload = result.expect_err("the panic reaches the caller");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"task 37 failed"));
        assert!(seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(counts(4, 1_000).iter().all(|&c| c == 1));
    }

    /// 8 threads each call `run` 1,000 times at once: every task of every call runs once.
    #[test]
    fn pool_concurrent_callers() {
        let _serial = serial();
        std::thread::scope(|s| {
            for caller in 0..8 {
                s.spawn(move || {
                    for i in 0..1_000 {
                        let tasks = 1 + (caller * 7 + i) % 40;
                        let c = counts(4, tasks);
                        assert_eq!(c.len(), tasks);
                        assert!(c.iter().all(|&n| n == 1), "caller {caller} call {i}");
                    }
                });
            }
        });
    }

    /// A `run` inside a task runs all of its tasks inline, on the task's own thread, as
    /// worker 0.
    #[test]
    fn pool_nested_call_runs_inline() {
        let _serial = serial();
        let outer_seen: Vec<AtomicUsize> = (0..8).map(|_| AtomicUsize::new(0)).collect();
        run(4, 8, &|outer: Task| {
            let me = std::thread::current().id();
            let inner_seen: Vec<AtomicUsize> = (0..10).map(|_| AtomicUsize::new(0)).collect();
            run(4, 10, &|inner: Task| {
                assert_eq!(inner.worker, 0);
                assert_eq!(std::thread::current().id(), me);
                inner_seen[inner.index].fetch_add(1, Ordering::Relaxed);
            });
            assert!(inner_seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
            outer_seen[outer.index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(outer_seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// The pool cannot start any worker: `run`, and a `run` nested inside one of its
    /// tasks, still finish with every task run once on the calling thread. Runs
    /// `spawn_failure_child` in a fresh process (where no worker exists yet) and fails if it
    /// fails or does not finish within 30 seconds.
    #[test]
    fn pool_spawn_failure_runs_inline() {
        let exe = std::env::current_exe().expect("the test binary's path");
        let mut child = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "--test-threads=1",
                "kernels::pool::tests::spawn_failure_child",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("the test binary runs");
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait().expect("waiting for the child") {
                assert!(status.success(), "the child failed: {status}");
                return;
            }
            if start.elapsed() > std::time::Duration::from_secs(30) {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the child did not finish within 30 s: run deadlocked");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Helper for `pool_spawn_failure_runs_inline`; ignored by default and run in a child
    /// process, since it disables worker spawning for the whole process.
    #[test]
    #[ignore = "run in a child process by pool_spawn_failure_runs_inline"]
    fn spawn_failure_child() {
        FAIL_SPAWN.store(true, Ordering::Relaxed);
        let outer_seen: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        run(4, 4, &|outer: Task| {
            assert_eq!(outer.worker, 0);
            let inner = counts(4, 4);
            assert!(inner.iter().all(|&c| c == 1));
            outer_seen[outer.index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(outer_seen.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    /// Prints the median time of an empty 4-task `run` on 4 threads, over 1,000 calls. The
    /// figure to quote comes from a run on its own, in release mode:
    /// `cargo test -p cavitation-core --release --lib kernels::pool::tests::pool_round_trip_time -- --exact
    /// --nocapture --test-threads=1`.
    #[test]
    fn pool_round_trip_time() {
        let _serial = serial();
        let empty = |_: Task| {};
        for _ in 0..100 {
            run(4, 4, &empty);
        }
        let mut micros: Vec<f64> = (0..1_000)
            .map(|_| {
                let start = Instant::now();
                run(4, 4, &empty);
                start.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        micros.sort_by(f64::total_cmp);
        println!(
            "pool round trip, 4 threads, empty job: median {:.2} us over 1000 calls",
            micros[500]
        );
    }
}
