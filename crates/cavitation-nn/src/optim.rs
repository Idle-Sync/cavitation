//! Optimisers: SGD (with momentum and weight decay), Adam and AdamW, and the gradient
//! statistics they collect while they update.
//!
//! All arithmetic is done per element in `f64` and stored back as `T`, so an `f32` run
//! rounds once per stored value. Momentum and Adam's moment estimates are stored as `T`, with
//! any value smaller than the type's smallest normal number stored as 0: left to decay, they
//! would pass through the subnormal range, where each update runs many times slower.
//!
//! Each parameter is updated in chunks of [`CHUNK`] elements, and the chunks of every
//! parameter of a step go to the kernel thread pool ([`threads`]) in one call, so a network
//! of many small parameters still keeps every thread busy. Each chunk keeps the gradient's
//! sum of squares in [`LANES`] partial sums, and a parameter's chunks are combined in
//! order. Every element gets exactly the arithmetic of the
//! plain loop, so the parameters and the optimiser's state are bit-identical to it and the
//! same for any thread count; only the order of the statistics' sums differs from the plain
//! loop, and it too is the same for any thread count. At [`Level::Reference`] the plain
//! loop runs instead, on the calling thread.

use cavitation_core::kernels::{Level, level, pool, threads};
use cavitation_core::{ParamId, ParamStore, Real};

/// Elements per chunk of a parameter in an update: 16,384 (64 KB of `f32` per array), small
/// enough that the 1M network's largest weight (802,816 elements) splits into 49 chunks to
/// share between threads, large enough that a chunk's work dwarfs handing it out.
const CHUNK: usize = 1 << 14;

/// Partial sums of the gradient's squares per chunk: independent sums the compiler can keep
/// in vector registers, instead of one chain of dependent adds.
const LANES: usize = 8;

/// `x` stored as a `T` for optimiser state, with any value below the type's smallest normal
/// number ([`Real::MIN_POSITIVE`]) stored as 0. Momentum and Adam's moments decay towards
/// zero, through the subnormal range, where reading and writing a value takes many times
/// longer; flushing it keeps every update at full speed. The test is on the `f64` value, a
/// compare and a select, so the result does not depend on the thread count.
#[inline(always)]
fn store_state<T: Real>(x: f64) -> T {
    if x.abs() < T::MIN_POSITIVE.to_f64() {
        T::ZERO
    } else {
        T::from_f64(x)
    }
}

/// Which loops an update runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// The plain per-element loop over each parameter, on the calling thread: the reference
    /// the chunked loops are tested against.
    Reference,
    /// Chunks of [`CHUNK`] elements on up to this many threads.
    Chunked(usize),
}

impl Path {
    /// The path for the process-wide settings: the plain loop at [`Level::Reference`], else
    /// chunks on [`threads`] threads.
    ///
    /// # Panics
    ///
    /// Panics, as [`level`] and [`threads`] do, if `CAVITATION_KERNEL` or
    /// `CAVITATION_THREADS` is invalid.
    fn current() -> Path {
        if level() == Level::Reference {
            Path::Reference
        } else {
            Path::Chunked(threads())
        }
    }
}

/// One chunk's gradient statistics.
#[derive(Clone, Copy, Debug, Default)]
struct ChunkStats {
    /// The sum of the squares of the chunk's gradient values.
    sum_squares: f64,
    /// Whether every one of them was finite.
    finite: bool,
}

/// The running statistics of one chunk: [`LANES`] partial sums, element `i` of the chunk
/// going to sum `i % LANES`.
struct Lanes {
    /// The partial sums of squares.
    sums: [f64; LANES],
    /// Whether every value so far was finite.
    finite: bool,
}

impl Lanes {
    /// No values yet.
    fn new() -> Lanes {
        Lanes {
            sums: [0.0; LANES],
            finite: true,
        }
    }

    /// Adds the gradient value `g` to partial sum `lane`.
    #[inline(always)]
    fn add(&mut self, lane: usize, g: f64) {
        self.sums[lane] += g * g;
        self.finite &= g.is_finite();
    }

    /// The chunk's statistics, the partial sums added in a fixed tree order.
    fn finish(&self) -> ChunkStats {
        let s = &self.sums;
        ChunkStats {
            sum_squares: ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7])),
            finite: self.finite,
        }
    }
}

/// Runs `f` on every element of one chunk, in order: `f` gets the element of each mutable
/// slice and the gradient value in `f64`, and the gradient goes into the chunk's
/// statistics. All slices have the gradient's length (checked by [`Parts::push`]).
/// The elements go in blocks of [`LANES`], each block's lanes written out with fixed
/// indices so the compiler can work on a block's lanes in vector registers.
#[inline(always)]
fn sweep<T: Real, const M: usize>(
    muts: [&mut [T]; M],
    grads: &[T],
    mut f: impl FnMut([&mut T; M], f64),
) -> ChunkStats {
    let (blocks, tail) = grads.as_chunks::<LANES>();
    let mut parts = muts.map(|m| m.as_chunks_mut::<LANES>());
    let mut lanes = Lanes::new();
    for (b, gs) in blocks.iter().enumerate() {
        let mut block = parts.each_mut().map(|(full, _)| &mut full[b]);
        for (lane, &g) in gs.iter().enumerate() {
            let g = g.to_f64();
            lanes.add(lane, g);
            f(block.each_mut().map(|x| &mut x[lane]), g);
        }
    }
    let mut rest = parts.map(|(_, rest)| rest);
    for (lane, &g) in tail.iter().enumerate() {
        let g = g.to_f64();
        lanes.add(lane, g);
        f(rest.each_mut().map(|x| &mut x[lane]), g);
    }
    lanes.finish()
}

/// One parameter of a chunked step: raw pointers to its `M` mutable buffers (the value
/// first) and to its gradient, all `len` elements long, the index of its first chunk among
/// the step's chunks, and the per-parameter constants `c` its chunks are updated with. A
/// parameter left out of the step (no gradient) is a part of length 0, with no chunks; its
/// pointers are never read.
#[derive(Clone, Copy)]
struct Part<T, C, const M: usize> {
    /// The parameter, for its statistics.
    id: ParamId,
    /// The mutable buffers, the value first.
    muts: [*mut T; M],
    /// The gradient.
    grads: *const T,
    /// The length of every buffer.
    len: usize,
    /// The index of the parameter's first chunk among the step's chunks.
    first_chunk: usize,
    /// The constants the parameter's chunks are updated with.
    c: C,
}

/// The parts of one chunked step. The list is cleared and filled at the start of a step
/// and cleared again when the step returns, so its pointers are only read during the step
/// that took them (a step that panics may leave some behind, which the next step clears
/// before reading any); it keeps its capacity, so steps after the first allocate nothing.
struct Parts<T, C, const M: usize>(Vec<Part<T, C, M>>);

impl<T, C, const M: usize> Parts<T, C, M> {
    /// An empty list.
    fn new() -> Self {
        Parts(Vec::new())
    }

    /// Adds parameter `id` with its `M` mutable buffers and its gradient (`None`: left out
    /// of the step, a part with no chunks), all of the gradient's length.
    fn push(&mut self, id: ParamId, muts: [&mut [T]; M], grads: Option<&[T]>, c: C) {
        let first_chunk = self
            .0
            .last()
            .map_or(0, |p| p.first_chunk + p.len.div_ceil(CHUNK));
        let len = grads.map_or(0, <[T]>::len);
        for (k, m) in muts.iter().enumerate() {
            assert!(
                grads.is_none() || m.len() == len,
                "optimiser buffer {k} has {} elements, the gradient {len}",
                m.len()
            );
        }
        self.0.push(Part {
            id,
            muts: muts.map(<[T]>::as_mut_ptr),
            grads: grads.map_or(std::ptr::null(), <[T]>::as_ptr),
            len,
            first_chunk,
            c,
        });
    }

    /// The number of chunks of all parts.
    fn chunks(&self) -> usize {
        self.0
            .last()
            .map_or(0, |p| p.first_chunk + p.len.div_ceil(CHUNK))
    }
}

impl<T, C, const M: usize> Clone for Parts<T, C, M> {
    /// An empty list: between steps there is nothing to copy.
    fn clone(&self) -> Self {
        Parts::new()
    }
}

impl<T, C, const M: usize> std::fmt::Debug for Parts<T, C, M> {
    /// The number of parts (0 between steps).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parts({})", self.0.len())
    }
}

// SAFETY: a part's pointers are only read during the step that took them (every step
// clears the list before filling it); during a step the list is used only by the thread
// that filled it and, through `run_parts`, by tasks that each form slices of disjoint
// chunks while the buffers are borrowed for the step.
unsafe impl<T: Send, C: Send, const M: usize> Send for Parts<T, C, M> {}

/// A shared reference that tasks on several threads may use: the parts of a step and the
/// chunk statistics' slots.
struct ForTasks<P>(P);

// SAFETY: `run_parts` uses it only to read parts that do not change during the call and to
// form references to disjoint buffer chunks and statistics slots, one task per chunk,
// while the buffers are borrowed for the step; sending those to another thread is sound
// when `T: Send`.
unsafe impl<P> Sync for ForTasks<P> {}

/// The update of one chunk: the parameter's constants and the chunk's parts of its `M`
/// mutable buffers and of its gradient in, the chunk's statistics out.
type ChunkUpdate<'a, T, C, const M: usize> =
    dyn Fn(&C, [&mut [T]; M], &[T]) -> ChunkStats + Sync + 'a;

/// Updates every part in chunks of [`CHUNK`] (chunk `i` of a part covers its elements
/// `i · CHUNK` onward) through one [`pool::run`] on up to `threads` threads, `chunk`
/// updating one chunk, then pushes each part's gradient sum of squares (its chunks' sums
/// added in order) and finiteness onto `stats`, in the order of the parts, and clears
/// `parts`. The chunks of a parameter and the order its sums are added in depend only on
/// its length, so the results are the same for any thread count and the same as updating
/// one parameter at a time. `partials` holds one slot per chunk; the optimisers size it on
/// their first step for every parameter's chunks, so it never grows after that.
fn run_parts<T: Real, C: Sync, const M: usize>(
    threads: usize,
    parts: &mut Parts<T, C, M>,
    partials: &mut Vec<ChunkStats>,
    stats: &mut GradStats,
    chunk: &ChunkUpdate<'_, T, C, M>,
) {
    let total = parts.chunks();
    if partials.len() < total {
        partials.resize(total, ChunkStats::default());
    }
    let list = ForTasks(&parts.0[..]);
    let slots = ForTasks(partials.as_mut_ptr());
    pool::run(threads, total, &|t: pool::Task| {
        let list = &list;
        let slots = &slots;
        // The last part that starts at or before this chunk; parts of length 0 start where
        // the next part does, so this is the part that holds the chunk.
        let p = &list.0[list.0.partition_point(|p| p.first_chunk <= t.index) - 1];
        let start = (t.index - p.first_chunk) * CHUNK;
        let n = CHUNK.min(p.len - start);
        let muts = p.muts.map(|ptr| {
            // SAFETY: each buffer holds `p.len` elements and `start + n <= p.len`; the
            // buffers are borrowed mutably for the step, distinct parameters' buffers do
            // not overlap, and `pool::run` gives each chunk index to one task only, so no
            // two slices in use overlap.
            unsafe { std::slice::from_raw_parts_mut(ptr.add(start), n) }
        });
        // SAFETY: as above, for the gradient, which is only read.
        let grads = unsafe { std::slice::from_raw_parts(p.grads.add(start), n) };
        // SAFETY: `t.index < total <= partials.len()`, and only this task uses this slot.
        unsafe { *slots.0.add(t.index) = chunk(&p.c, muts, grads) };
    });
    for p in &parts.0 {
        let first = p.first_chunk;
        let (sum, finite) = partials[first..first + p.len.div_ceil(CHUNK)]
            .iter()
            .fold((0.0, true), |(sum, finite), c| {
                (sum + c.sum_squares, finite & c.finite)
            });
        stats.push(p.id, sum, finite);
    }
    parts.0.clear();
}

/// What the optimiser saw in the gradients during a step: for every parameter the sum of the
/// squares of its gradient (in `f64`) and whether every value was finite.
///
/// The optimiser fills it in the same pass that updates the parameters, so reading the gradients
/// costs no extra pass. A parameter whose gradient is stale (nothing contributed to it since
/// [`ParamStore::zero_grad`]) is not updated and counts as zeros here. A `GradStats` can be reused
/// from step to step; after the first step it allocates nothing.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{GradStats, Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::zeros(&[2]));
/// store.grads.accumulate(p).0.data_mut().copy_from_slice(&[3.0, 4.0]);
/// let mut stats = GradStats::new();
/// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
/// assert_eq!(stats.norm(p), 5.0);
/// assert!(stats.all_finite());
/// ```
#[derive(Clone, Debug, Default)]
pub struct GradStats {
    /// The parameters seen in the last step, in the order of the store.
    ids: Vec<ParamId>,
    /// The sum of squares of each parameter's gradient.
    sum_squares: Vec<f64>,
    /// Whether every gradient value of each parameter was finite.
    finite: Vec<bool>,
}

impl GradStats {
    /// Empty statistics, to be filled by [`Optimizer::step_with_stats`].
    ///
    /// ```
    /// let stats = cavitation_nn::GradStats::new();
    /// assert!(stats.all_finite());
    /// ```
    pub fn new() -> GradStats {
        GradStats::default()
    }

    /// Forgets the last step and makes room for `n` parameters.
    fn reset(&mut self, n: usize) {
        self.ids.clear();
        self.sum_squares.clear();
        self.finite.clear();
        self.ids.reserve(n);
        self.sum_squares.reserve(n);
        self.finite.reserve(n);
    }

    /// Records one parameter.
    fn push(&mut self, id: ParamId, sum_squares: f64, finite: bool) {
        self.ids.push(id);
        self.sum_squares.push(sum_squares);
        self.finite.push(finite);
    }

    /// The Euclidean norm of the gradient of `id` in the last step: the square root of its
    /// sum of squares.
    ///
    /// # Panics
    ///
    /// Panics if `id` was not seen in the last step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(p).0.data_mut()[0] = -2.0;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert_eq!(stats.norm(p), 2.0);
    /// ```
    pub fn norm(&self, id: ParamId) -> f64 {
        match self.ids.iter().position(|&i| i == id) {
            Some(k) => self.sum_squares[k].sqrt(),
            None => panic!("GradStats::norm: {id:?} was not seen in the last step"),
        }
    }

    /// Whether every gradient value of every parameter was finite.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(p).0.data_mut()[0] = f32::NAN;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert!(!stats.all_finite());
    /// ```
    pub fn all_finite(&self) -> bool {
        self.finite.iter().all(|&f| f)
    }

    /// The first parameter, in the order of the store, whose gradient held a NaN or an
    /// infinity; `None` when all were finite.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let a = store.add(Tensor::zeros(&[1]));
    /// let b = store.add(Tensor::zeros(&[1]));
    /// store.grads.accumulate(b).0.data_mut()[0] = f32::INFINITY;
    /// let mut stats = GradStats::new();
    /// Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
    /// assert_eq!(stats.first_non_finite(), Some(b));
    /// # let _ = a;
    /// ```
    pub fn first_non_finite(&self) -> Option<ParamId> {
        self.finite.iter().position(|&f| !f).map(|k| self.ids[k])
    }
}

/// Changes parameters from their gradients.
///
/// An optimiser keeps its state (momentum, moment estimates) in buffers it allocates on the first
/// step and reuses afterwards. A parameter whose gradient is stale (nothing has contributed to it
/// since [`ParamStore::zero_grad`]) is skipped, as PyTorch skips a parameter whose `grad` is
/// `None`: its value and its state stay as they are, and Adam's bias correction counts only the
/// steps that updated it. The parameter store must not change its number or the sizes of its
/// parameters after that: a later step panics if it does.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 0.5;
/// let mut opt = Sgd::new(0.1);
/// opt.step(&mut store);
/// assert_eq!(store.value(p).data(), &[0.95]);
/// ```
pub trait Optimizer<T: Real>: Send {
    /// Updates every parameter that has a gradient (see [`Optimizer`] for those without one)
    /// and fills `stats` with the gradients' sums of squares and finiteness, in the same
    /// pass.
    ///
    /// # Panics
    ///
    /// Panics if the store's parameters differ in number or size from the first step, and
    /// on first use, as [`level`] and [`threads`] do, if `CAVITATION_KERNEL` or
    /// `CAVITATION_THREADS` is invalid.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{GradStats, Optimizer, Adam};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[2], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut().fill(1.0);
    /// let mut stats = GradStats::new();
    /// Adam::new(0.01).step_with_stats(&mut store, &mut stats);
    /// assert!((stats.norm(p) - 2f64.sqrt()).abs() < 1e-12);
    /// ```
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats);

    /// Updates every parameter that has a gradient, as
    /// [`step_with_stats`](Optimizer::step_with_stats) does. The statistics go into a [`GradStats`]
    /// the optimiser owns, allocated on the first step and reused, so after the first step this
    /// allocates nothing; read them with [`last_stats`](Optimizer::last_stats).
    ///
    /// # Panics
    ///
    /// Panics if the store's parameters differ in number or size from the first step.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// store.grads.accumulate(p).0.data_mut()[0] = 1.0;
    /// let mut opt = Sgd::new(0.5);
    /// opt.step(&mut store);
    /// assert_eq!(store.value(p).data(), &[0.5]);
    /// assert_eq!(opt.last_stats().norm(p), 1.0);
    /// ```
    fn step(&mut self, params: &mut ParamStore<T>);

    /// The statistics of the last [`step`](Optimizer::step) (empty before the first). A step
    /// taken with [`step_with_stats`](Optimizer::step_with_stats) fills the caller's
    /// `GradStats` instead and leaves this one unchanged.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Adam, Optimizer};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::zeros(&[2]));
    /// store.grads.accumulate(p).0.data_mut().copy_from_slice(&[3.0, 4.0]);
    /// let mut opt = Adam::new(0.01);
    /// opt.step(&mut store);
    /// assert_eq!(opt.last_stats().norm(p), 5.0);
    /// ```
    fn last_stats(&self) -> &GradStats;

    /// The current learning rate.
    ///
    /// ```
    /// use cavitation_nn::{Optimizer, Sgd};
    /// assert_eq!(Optimizer::<f32>::learning_rate(&Sgd::new(0.1)), 0.1);
    /// ```
    fn learning_rate(&self) -> f64;

    /// Sets the learning rate used by later steps.
    ///
    /// ```
    /// use cavitation_nn::{Optimizer, Sgd};
    /// let mut opt = Sgd::new(0.1);
    /// Optimizer::<f32>::set_learning_rate(&mut opt, 0.01);
    /// assert_eq!(Optimizer::<f32>::learning_rate(&opt), 0.01);
    /// ```
    fn set_learning_rate(&mut self, lr: f64);
}

impl<T: Real, O: Optimizer<T> + ?Sized> Optimizer<T> for Box<O> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        (**self).step_with_stats(params, stats);
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        (**self).step(params);
    }

    fn last_stats(&self) -> &GradStats {
        (**self).last_stats()
    }

    fn learning_rate(&self) -> f64 {
        (**self).learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        (**self).set_learning_rate(lr);
    }
}

/// The parameter sizes seen on the first step, checked on every later step.
#[derive(Clone, Debug, Default)]
struct Sizes {
    /// Whether the first step has happened.
    seen: bool,
    /// The element count of each parameter, in the order of the store.
    lens: Vec<usize>,
}

impl Sizes {
    /// Records the sizes on the first call (returns `true`); on later calls panics unless the
    /// store has the same number of parameters of the same sizes (returns `false`). Allocates
    /// only on the first call.
    fn prepare<T: Real>(&mut self, name: &str, params: &mut ParamStore<T>) -> bool {
        if !self.seen {
            self.lens
                .extend(params.iter_mut().map(|(_, v, _)| v.data().len()));
            self.seen = true;
            return true;
        }
        let now = params.len();
        assert!(
            now == self.lens.len(),
            "{name}: the store has {now} parameters but the first step saw {}; \
             an optimiser's state is sized on its first step",
            self.lens.len()
        );
        for (k, (_, v, _)) in params.iter_mut().enumerate() {
            assert!(
                v.data().len() == self.lens[k],
                "{name}: parameter {k} has {} elements but the first step saw {}; \
                 an optimiser's state is sized on its first step",
                v.data().len(),
                self.lens[k]
            );
        }
        false
    }

    /// The number of chunks of [`CHUNK`] in a step that updates every parameter: the most
    /// [`run_parts`] needs statistics slots for.
    fn chunks(&self) -> usize {
        self.lens.iter().map(|n| n.div_ceil(CHUNK)).sum()
    }
}

/// Stochastic gradient descent with optional momentum and weight decay, with PyTorch's
/// `torch.optim.SGD` semantics (dampening 0, no Nesterov).
///
/// With weight decay `wd` the gradient becomes `g + wd·p`. With momentum `μ` the buffer is
/// `b ← μ·b + g` (`b = g` on the first step) and the parameter moves by `−lr·b`.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Optimizer, Sgd};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// let mut opt = Sgd::new(0.1).with_momentum(0.9);
/// for _ in 0..2 {
///     store.zero_grad();
///     store.grads.accumulate(p).0.data_mut()[0] = 1.0;
///     opt.step(&mut store);
/// }
/// // b = 1, then b = 1.9: 1 - 0.1 - 0.19
/// assert!((store.value(p).data()[0] - 0.71).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Sgd<T: Real = f32> {
    /// The learning rate.
    lr: f64,
    /// The momentum factor; 0 disables the buffer.
    momentum: f64,
    /// The weight decay added to the gradient.
    weight_decay: f64,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The momentum buffer of each parameter (empty without momentum).
    buffers: Vec<Vec<T>>,
    /// Whether each parameter's momentum buffer holds a value: false until its first update,
    /// which sets the buffer to that step's gradient, as PyTorch starts its buffer (empty
    /// without momentum).
    started: Vec<bool>,
    /// The statistics of the last `step`.
    stats: GradStats,
    /// One slot per chunk of a step, for the chunks' statistics.
    partials: Vec<ChunkStats>,
    /// The parameters of a chunked step without momentum (empty between steps).
    plain: Parts<T, (), 1>,
    /// The parameters of a chunked step with momentum, each with whether it is the
    /// parameter's first update (empty between steps).
    with_momentum: Parts<T, bool, 2>,
}

impl<T: Real> Sgd<T> {
    /// Plain gradient descent with learning rate `lr`.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> Sgd<T> {
        Sgd {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            sizes: Sizes::default(),
            buffers: Vec::new(),
            started: Vec::new(),
            stats: GradStats::new(),
            partials: Vec::new(),
            plain: Parts::new(),
            with_momentum: Parts::new(),
        }
    }

    /// Sets the momentum factor.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1).with_momentum(0.9);
    /// # let _ = opt;
    /// ```
    pub fn with_momentum(mut self, momentum: f64) -> Sgd<T> {
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p`.
    ///
    /// ```
    /// let opt: cavitation_nn::Sgd = cavitation_nn::Sgd::new(0.1).with_weight_decay(1e-4);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Sgd<T> {
        self.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> Sgd<T> {
    /// The momentum buffer of parameter `id`, or `None` while there is none: the optimiser
    /// has no momentum, has not stepped yet, or has not updated this parameter yet. The ids
    /// are those of the store the optimiser steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Optimizer, Sgd};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut opt = Sgd::new(0.1).with_momentum(0.9);
    /// assert!(opt.momentum(p).is_none());
    /// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
    /// opt.step(&mut store);
    /// assert_eq!(opt.momentum(p), Some(&[2.0f32][..]));
    /// ```
    pub fn momentum(&self, id: ParamId) -> Option<&[T]> {
        let k = id.position();
        (*self.started.get(k)?).then(|| &self.buffers[k][..])
    }

    /// One step on `path`: [`Optimizer::step_with_stats`] with the loops chosen explicitly.
    fn update(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats, path: Path) {
        if self.sizes.prepare("Sgd", params) {
            self.partials = vec![ChunkStats::default(); self.sizes.chunks()];
            if self.momentum != 0.0 {
                self.buffers = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
                self.started = vec![false; self.buffers.len()];
            }
        }
        stats.reset(params.len());
        let (lr, mu, wd) = (self.lr, self.momentum, self.weight_decay);
        if let Path::Chunked(threads) = path {
            return self.update_chunked(threads, params, stats);
        }
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let Some(grad) = grad else {
                // No gradient since `zero_grad`: skipped, as PyTorch skips `grad is None`.
                stats.push(id, 0.0, true);
                continue;
            };
            let values = value.data_mut();
            let grads = grad.data();
            let (buffer, first) = if mu != 0.0 {
                let first = !std::mem::replace(&mut self.started[k], true);
                (&mut self.buffers[k][..], first)
            } else {
                (&mut [][..], false)
            };
            let (sum, finite) = sgd_reference(lr, mu, wd, first, values, buffer, grads);
            stats.push(id, sum, finite);
        }
    }

    /// One step in chunks of every parameter at once, on up to `threads` threads: the
    /// arithmetic of [`sgd_reference`] per element.
    fn update_chunked(
        &mut self,
        threads: usize,
        params: &mut ParamStore<T>,
        stats: &mut GradStats,
    ) {
        let (lr, mu, wd) = (self.lr, self.momentum, self.weight_decay);
        // A step that panicked may have left parts behind.
        self.plain.0.clear();
        self.with_momentum.0.clear();
        let parts = &mut self.partials;
        if mu == 0.0 {
            for (id, value, grad) in params.iter_mut() {
                // A parameter with no gradient since `zero_grad` is skipped, as PyTorch
                // skips `grad is None`: a part with no chunks.
                let grads = grad.as_ref().map(|g| g.data());
                self.plain.push(id, [value.data_mut()], grads, ());
            }
            let list = &mut self.plain;
            if wd == 0.0 {
                run_parts(threads, list, parts, stats, &|_, [v], g| {
                    sgd_chunk::<T, false>(lr, wd, v, g)
                });
            } else {
                run_parts(threads, list, parts, stats, &|_, [v], g| {
                    sgd_chunk::<T, true>(lr, wd, v, g)
                });
            }
            return;
        }
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let grads = grad.as_ref().map(|g| g.data());
            // Only an update starts the buffer; a skipped parameter keeps its state.
            let first = grads.is_some() && !std::mem::replace(&mut self.started[k], true);
            let buffer = &mut self.buffers[k][..];
            self.with_momentum
                .push(id, [value.data_mut(), buffer], grads, first);
        }
        let list = &mut self.with_momentum;
        if wd == 0.0 {
            run_parts(threads, list, parts, stats, &|&first, [v, b], g| {
                if first {
                    sgd_momentum_chunk::<T, false, true>(lr, mu, wd, v, b, g)
                } else {
                    sgd_momentum_chunk::<T, false, false>(lr, mu, wd, v, b, g)
                }
            });
        } else {
            run_parts(threads, list, parts, stats, &|&first, [v, b], g| {
                if first {
                    sgd_momentum_chunk::<T, true, true>(lr, mu, wd, v, b, g)
                } else {
                    sgd_momentum_chunk::<T, true, false>(lr, mu, wd, v, b, g)
                }
            });
        }
    }
}

/// The plain SGD loop over one parameter, element by element on the calling thread: the
/// reference for [`sgd_chunk`] and [`sgd_momentum_chunk`]. `buffer` is the momentum buffer
/// (empty when `mu` is 0) and `first` whether this is the parameter's first update. Returns the
/// gradient's sum of squares and whether it was all finite.
fn sgd_reference<T: Real>(
    lr: f64,
    mu: f64,
    wd: f64,
    first: bool,
    values: &mut [T],
    buffer: &mut [T],
    grads: &[T],
) -> (f64, bool) {
    let (mut sum, mut finite) = (0.0f64, true);
    for i in 0..values.len() {
        let g = grads[i].to_f64();
        sum += g * g;
        finite &= g.is_finite();
        let p = values[i].to_f64();
        let mut step = g;
        if wd != 0.0 {
            step += wd * p;
        }
        if mu != 0.0 {
            let buf = &mut buffer[i];
            step = if first {
                step
            } else {
                mu * buf.to_f64() + step
            };
            *buf = store_state(step);
        }
        values[i] = T::from_f64(p - lr * step);
    }
    (sum, finite)
}

/// One chunk of SGD without momentum, with weight decay when `WD`: the arithmetic of
/// [`sgd_reference`] per element.
fn sgd_chunk<T: Real, const WD: bool>(
    lr: f64,
    wd: f64,
    values: &mut [T],
    grads: &[T],
) -> ChunkStats {
    sweep([values], grads, |[value], g| {
        let p = value.to_f64();
        let mut step = g;
        if WD {
            step += wd * p;
        }
        *value = T::from_f64(p - lr * step);
    })
}

/// One chunk of SGD with momentum, with weight decay when `WD`, on the parameter's first
/// update when `FIRST`: the arithmetic of [`sgd_reference`] per element.
fn sgd_momentum_chunk<T: Real, const WD: bool, const FIRST: bool>(
    lr: f64,
    mu: f64,
    wd: f64,
    values: &mut [T],
    buffer: &mut [T],
    grads: &[T],
) -> ChunkStats {
    sweep([values, buffer], grads, |[value, buf], g| {
        let p = value.to_f64();
        let mut step = g;
        if WD {
            step += wd * p;
        }
        step = if FIRST {
            step
        } else {
            mu * buf.to_f64() + step
        };
        *buf = store_state(step);
        *value = T::from_f64(p - lr * step);
    })
}

impl<T: Real> Optimizer<T> for Sgd<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.update(params, stats, Path::current());
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.stats);
        self.step_with_stats(params, &mut stats);
        self.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.stats
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// The state and update shared by [`Adam`] and [`AdamW`].
#[derive(Clone, Debug)]
struct AdamCore<T: Real> {
    /// The learning rate.
    lr: f64,
    /// The decay rate of the first moment.
    beta1: f64,
    /// The decay rate of the second moment.
    beta2: f64,
    /// Added to the square root of the second moment, outside the root.
    eps: f64,
    /// The weight decay.
    weight_decay: f64,
    /// The updates each parameter has had so far, for its bias corrections: a parameter
    /// skipped for want of a gradient does not count the step, as in PyTorch's per-parameter
    /// `state["step"]`.
    steps: Vec<i32>,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The first moment of each parameter.
    m: Vec<Vec<T>>,
    /// The second moment of each parameter.
    v: Vec<Vec<T>>,
    /// The statistics of the last `step`.
    stats: GradStats,
    /// One slot per chunk of a step, for the chunks' statistics.
    partials: Vec<ChunkStats>,
    /// The parameters of a chunked step, each with its step's constants (empty between
    /// steps).
    parts: Parts<T, AdamStep, 3>,
}

impl<T: Real> AdamCore<T> {
    /// The defaults of `torch.optim.Adam` with learning rate `lr` and weight decay `wd`.
    fn new(lr: f64, weight_decay: f64) -> AdamCore<T> {
        AdamCore {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            steps: Vec::new(),
            sizes: Sizes::default(),
            m: Vec::new(),
            v: Vec::new(),
            stats: GradStats::new(),
            partials: Vec::new(),
            parts: Parts::new(),
        }
    }

    /// The two moments of parameter `id`, or `None` until it has been updated once.
    fn moments(&self, id: ParamId) -> Option<(&[T], &[T])> {
        let k = id.position();
        (*self.steps.get(k)? > 0).then(|| (&self.m[k][..], &self.v[k][..]))
    }

    /// Checks the decay rates.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    fn set_betas(&mut self, beta1: f64, beta2: f64) {
        assert!(
            (0.0..1.0).contains(&beta1) && (0.0..1.0).contains(&beta2),
            "Adam betas must be in [0, 1), got ({beta1}, {beta2})"
        );
        self.beta1 = beta1;
        self.beta2 = beta2;
    }

    /// One step on `path`. `decoupled` selects AdamW's decay (`p ← p·(1 − lr·wd)` before
    /// the update) over Adam's (`wd·p` added to the gradient).
    fn step(
        &mut self,
        name: &str,
        decoupled: bool,
        params: &mut ParamStore<T>,
        stats: &mut GradStats,
        path: Path,
    ) {
        if self.sizes.prepare(name, params) {
            self.m = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
            self.v = self.sizes.lens.iter().map(|&n| vec![T::ZERO; n]).collect();
            self.steps = vec![0; self.m.len()];
            self.partials = vec![ChunkStats::default(); self.sizes.chunks()];
        }
        stats.reset(params.len());
        // A step that panicked may have left parts behind.
        self.parts.0.clear();
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let Some(grad) = grad else {
                // No gradient since `zero_grad`: skipped, as PyTorch skips `grad is None`
                // (in a chunked step, a part with no chunks).
                match path {
                    Path::Reference => stats.push(id, 0.0, true),
                    Path::Chunked(_) => {
                        self.parts
                            .push(id, [&mut [], &mut [], &mut []], None, AdamStep::default())
                    }
                }
                continue;
            };
            self.steps[k] += 1;
            let c = self.constants(self.steps[k]);
            let values = value.data_mut();
            let grads = grad.data();
            let (m, v) = (&mut self.m[k][..], &mut self.v[k][..]);
            match path {
                Path::Reference => {
                    let (sum, finite) = adam_reference(&c, decoupled, values, m, v, grads);
                    stats.push(id, sum, finite);
                }
                Path::Chunked(_) => self.parts.push(id, [values, m, v], Some(grads), c),
            }
        }
        let Path::Chunked(threads) = path else {
            return;
        };
        let (list, parts) = (&mut self.parts, &mut self.partials);
        match (decoupled, self.weight_decay != 0.0) {
            (true, _) => run_parts(threads, list, parts, stats, &|c, [p, m, v], g| {
                adam_chunk::<T, true, false>(c, p, m, v, g)
            }),
            (false, true) => run_parts(threads, list, parts, stats, &|c, [p, m, v], g| {
                adam_chunk::<T, false, true>(c, p, m, v, g)
            }),
            (false, false) => run_parts(threads, list, parts, stats, &|c, [p, m, v], g| {
                adam_chunk::<T, false, false>(c, p, m, v, g)
            }),
        }
    }

    /// The constants of a parameter's update number `t` (from 1).
    fn constants(&self, t: i32) -> AdamStep {
        let (lr, b1, b2, wd) = (self.lr, self.beta1, self.beta2, self.weight_decay);
        AdamStep {
            b1,
            b2,
            eps: self.eps,
            wd,
            step_size: lr / (1.0 - b1.powi(t)),
            inv_sqrt_correction2: 1.0 / (1.0 - b2.powi(t)).sqrt(),
            shrink: 1.0 - lr * wd,
        }
    }
}

/// The constants of one Adam or AdamW step (all 0 for a parameter left out of a step).
#[derive(Clone, Copy, Debug, Default)]
struct AdamStep {
    /// The decay rate of the first moment.
    b1: f64,
    /// The decay rate of the second moment.
    b2: f64,
    /// Added to the square root of the second moment.
    eps: f64,
    /// The weight decay.
    wd: f64,
    /// The learning rate over the first moment's bias correction, `lr / (1 − β₁ᵗ)`.
    step_size: f64,
    /// One over the square root of the second moment's bias correction, `1 / √(1 − β₂ᵗ)`.
    inv_sqrt_correction2: f64,
    /// AdamW's decay factor, `1 − lr·wd`.
    shrink: f64,
}

/// The plain Adam loop over one parameter, element by element on the calling thread: the
/// reference for [`adam_chunk`]. Returns the gradient's sum of squares and whether it was
/// all finite.
fn adam_reference<T: Real>(
    c: &AdamStep,
    decoupled: bool,
    values: &mut [T],
    m: &mut [T],
    v: &mut [T],
    grads: &[T],
) -> (f64, bool) {
    let (mut sum, mut finite) = (0.0f64, true);
    for i in 0..values.len() {
        let raw = grads[i].to_f64();
        sum += raw * raw;
        finite &= raw.is_finite();
        let mut p = values[i].to_f64();
        let mut g = raw;
        if decoupled {
            p *= c.shrink;
        } else if c.wd != 0.0 {
            g += c.wd * p;
        }
        let mi = c.b1 * m[i].to_f64() + (1.0 - c.b1) * g;
        let vi = c.b2 * v[i].to_f64() + (1.0 - c.b2) * g * g;
        m[i] = store_state(mi);
        v[i] = store_state(vi);
        values[i] = T::from_f64(adam_update(c, p, mi, vi));
    }
    (sum, finite)
}

/// One chunk of Adam (AdamW's decoupled decay when `DECOUPLED`, Adam's decay added to the
/// gradient when `WD`): the arithmetic of [`adam_reference`] per element.
fn adam_chunk<T: Real, const DECOUPLED: bool, const WD: bool>(
    c: &AdamStep,
    values: &mut [T],
    m: &mut [T],
    v: &mut [T],
    grads: &[T],
) -> ChunkStats {
    sweep([values, m, v], grads, |[value, m, v], raw| {
        let mut p = value.to_f64();
        let mut g = raw;
        if DECOUPLED {
            p *= c.shrink;
        } else if WD {
            g += c.wd * p;
        }
        let mi = c.b1 * m.to_f64() + (1.0 - c.b1) * g;
        let vi = c.b2 * v.to_f64() + (1.0 - c.b2) * g * g;
        *m = store_state(mi);
        *v = store_state(vi);
        *value = T::from_f64(adam_update(c, p, mi, vi));
    })
}

/// The updated parameter from its value `p` and its new moments `m` and `v`:
/// `p − step_size · m / (√v / √(1 − β₂ᵗ) + eps)`, the bias corrections folded into two
/// scalars as `torch.optim.Adam` folds them, so each element takes one square root and
/// one division. (PyTorch divides `√v` by `√(1 − β₂ᵗ)`; this multiplies by its inverse,
/// which can differ in the last bit.)
#[inline(always)]
fn adam_update(c: &AdamStep, p: f64, m: f64, v: f64) -> f64 {
    p - c.step_size * m / (v.sqrt() * c.inv_sqrt_correction2 + c.eps)
}

/// Adam (Kingma and Ba, 2015) with PyTorch's `torch.optim.Adam` semantics: bias-corrected
/// moments, `eps` added outside the square root, and weight decay added to the gradient.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Adam, Optimizer};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 3.0;
/// Adam::new(0.01).step(&mut store);
/// // The first step moves by lr in the direction of the gradient.
/// assert!((store.value(p).data()[0] - 0.99).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Adam<T: Real = f32> {
    /// The shared state and update.
    core: AdamCore<T>,
}

impl<T: Real> Adam<T> {
    /// Adam with learning rate `lr`, betas (0.9, 0.999), eps 1e-8 and no weight decay.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> Adam<T> {
        Adam {
            core: AdamCore::new(lr, 0.0),
        }
    }

    /// Sets the decay rates of the two moments.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_betas(0.8, 0.99);
    /// # let _ = opt;
    /// ```
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Adam<T> {
        self.core.set_betas(beta1, beta2);
        self
    }

    /// Sets the constant added to the square root of the second moment.
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_eps(1e-6);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> Adam<T> {
        self.core.eps = eps;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p` (coupled, so it is scaled by
    /// the adaptive step like any gradient).
    ///
    /// ```
    /// let opt: cavitation_nn::Adam = cavitation_nn::Adam::new(0.001).with_weight_decay(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Adam<T> {
        self.core.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> Adam<T> {
    /// The first and second moment estimates of parameter `id`, or `None` until it has been
    /// updated once. The ids are those of the store the optimiser steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Adam, Optimizer};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut opt = Adam::new(0.01);
    /// assert!(opt.moments(p).is_none());
    /// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
    /// opt.step(&mut store);
    /// let (m, v) = opt.moments(p).unwrap();
    /// // m = (1 - 0.9)·2 and v = (1 - 0.999)·2².
    /// assert!((m[0] - 0.2).abs() < 1e-7 && (v[0] - 0.004).abs() < 1e-7);
    /// ```
    pub fn moments(&self, id: ParamId) -> Option<(&[T], &[T])> {
        self.core.moments(id)
    }
}

impl<T: Real> Optimizer<T> for Adam<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.core
            .step("Adam", false, params, stats, Path::current());
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.core.stats);
        self.step_with_stats(params, &mut stats);
        self.core.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.core.stats
    }

    fn learning_rate(&self) -> f64 {
        self.core.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.core.lr = lr;
    }
}

/// AdamW (Loshchilov and Hutter, 2019) with PyTorch's `torch.optim.AdamW` semantics: the
/// weight decay is decoupled, `p ← p·(1 − lr·wd)` before the Adam update, and the default
/// weight decay is 0.01.
///
/// The default decay pulls every parameter towards 0 on every step, so with a constant rate it
/// can keep a run from settling on a minimum away from 0; set `with_weight_decay(0.0)` when
/// the minimum is not near 0.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{AdamW, Optimizer};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// // A zero gradient: only the decay acts, 1 - 0.1 * 0.01 * 1.
/// store.grads.accumulate(p).0.data_mut()[0] = 0.0;
/// AdamW::new(0.1).step(&mut store);
/// assert!((store.value(p).data()[0] - 0.999).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct AdamW<T: Real = f32> {
    /// The shared state and update.
    core: AdamCore<T>,
}

impl<T: Real> AdamW<T> {
    /// AdamW with learning rate `lr`, betas (0.9, 0.999), eps 1e-8 and weight decay 0.01.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> AdamW<T> {
        AdamW {
            core: AdamCore::new(lr, 0.01),
        }
    }

    /// Sets the decay rates of the two moments.
    ///
    /// # Panics
    ///
    /// Panics unless both are in `[0, 1)`.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_betas(0.8, 0.99);
    /// # let _ = opt;
    /// ```
    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> AdamW<T> {
        self.core.set_betas(beta1, beta2);
        self
    }

    /// Sets the constant added to the square root of the second moment.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_eps(1e-6);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> AdamW<T> {
        self.core.eps = eps;
        self
    }

    /// Sets the decoupled weight decay.
    ///
    /// ```
    /// let opt: cavitation_nn::AdamW = cavitation_nn::AdamW::new(0.001).with_weight_decay(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> AdamW<T> {
        self.core.weight_decay = weight_decay;
        self
    }
}

impl<T: Real> AdamW<T> {
    /// The first and second moment estimates of parameter `id`, or `None` until it has been
    /// updated once. The ids are those of the store the optimiser steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{AdamW, Optimizer};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut opt = AdamW::new(0.01);
    /// assert!(opt.moments(p).is_none());
    /// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
    /// opt.step(&mut store);
    /// let (m, v) = opt.moments(p).unwrap();
    /// // m = (1 - 0.9)·2 and v = (1 - 0.999)·2².
    /// assert!((m[0] - 0.2).abs() < 1e-7 && (v[0] - 0.004).abs() < 1e-7);
    /// ```
    pub fn moments(&self, id: ParamId) -> Option<(&[T], &[T])> {
        self.core.moments(id)
    }
}

impl<T: Real> Optimizer<T> for AdamW<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.core
            .step("AdamW", true, params, stats, Path::current());
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.core.stats);
        self.step_with_stats(params, &mut stats);
        self.core.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.core.stats
    }

    fn learning_rate(&self) -> f64 {
        self.core.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.core.lr = lr;
    }
}

/// RMSprop (Hinton's lecture notes, 2012) with PyTorch's `torch.optim.RMSprop` semantics,
/// uncentred: a running average of the squared gradient, `v ← α·v + (1 − α)·g²`, divides
/// each step, `p ← p − lr·g / (√v + eps)`, with weight decay added to the gradient as
/// `wd·p` and, with momentum `μ`, a buffer `b ← μ·b + g / (√v + eps)` that the step takes
/// instead, `p ← p − lr·b`. The average and the buffer start at 0. Defaults as PyTorch's:
/// α 0.99, eps 1e-8, no momentum, no weight decay.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Optimizer, RmsProp};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
/// RmsProp::new(0.01).step(&mut store);
/// // v = 0.01·4, so the step is 0.01·2 / 0.2 = 0.1.
/// assert!((store.value(p).data()[0] - 0.9).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct RmsProp<T: Real = f32> {
    /// The learning rate.
    lr: f64,
    /// The decay rate of the squared-gradient average.
    alpha: f64,
    /// Added to the square root of the average, outside the root.
    eps: f64,
    /// The momentum factor; 0 disables the buffer.
    momentum: f64,
    /// The weight decay added to the gradient.
    weight_decay: f64,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The squared-gradient average of each parameter.
    square_avg: Vec<Vec<T>>,
    /// The momentum buffer of each parameter (empty without momentum).
    buffers: Vec<Vec<T>>,
    /// Whether each parameter has been updated: its state means something only then.
    started: Vec<bool>,
    /// The statistics of the last `step`.
    stats: GradStats,
    /// One slot per chunk of a step, for the chunks' statistics.
    partials: Vec<ChunkStats>,
    /// The parameters of a chunked step without momentum (empty between steps).
    plain: Parts<T, (), 2>,
    /// The parameters of a chunked step with momentum (empty between steps).
    with_momentum: Parts<T, (), 3>,
}

impl<T: Real> RmsProp<T> {
    /// RMSprop with learning rate `lr`, α 0.99, eps 1e-8, no momentum and no weight decay.
    ///
    /// ```
    /// let opt: cavitation_nn::RmsProp = cavitation_nn::RmsProp::new(0.01);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> RmsProp<T> {
        RmsProp {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            sizes: Sizes::default(),
            square_avg: Vec::new(),
            buffers: Vec::new(),
            started: Vec::new(),
            stats: GradStats::new(),
            partials: Vec::new(),
            plain: Parts::new(),
            with_momentum: Parts::new(),
        }
    }

    /// Sets the decay rate α of the squared-gradient average.
    ///
    /// # Panics
    ///
    /// Panics unless `alpha` is in `[0, 1]`.
    ///
    /// ```
    /// let opt: cavitation_nn::RmsProp = cavitation_nn::RmsProp::new(0.01).with_alpha(0.9);
    /// # let _ = opt;
    /// ```
    pub fn with_alpha(mut self, alpha: f64) -> RmsProp<T> {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "RmsProp alpha must be in [0, 1], got {alpha}"
        );
        self.alpha = alpha;
        self
    }

    /// Sets the constant added to the square root of the average.
    ///
    /// ```
    /// let opt: cavitation_nn::RmsProp = cavitation_nn::RmsProp::new(0.01).with_eps(1e-6);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> RmsProp<T> {
        self.eps = eps;
        self
    }

    /// Sets the momentum factor μ.
    ///
    /// ```
    /// let opt: cavitation_nn::RmsProp = cavitation_nn::RmsProp::new(0.01).with_momentum(0.9);
    /// # let _ = opt;
    /// ```
    pub fn with_momentum(mut self, momentum: f64) -> RmsProp<T> {
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p`.
    ///
    /// ```
    /// let opt: cavitation_nn::RmsProp = cavitation_nn::RmsProp::new(0.01).with_weight_decay(1e-4);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> RmsProp<T> {
        self.weight_decay = weight_decay;
        self
    }

    /// The squared-gradient average of parameter `id`, or `None` until it has been updated
    /// once. The ids are those of the store the optimiser steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Optimizer, RmsProp};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut opt = RmsProp::new(0.01);
    /// assert!(opt.square_avg(p).is_none());
    /// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
    /// opt.step(&mut store);
    /// assert!((opt.square_avg(p).unwrap()[0] - 0.04).abs() < 1e-7);
    /// ```
    pub fn square_avg(&self, id: ParamId) -> Option<&[T]> {
        let k = id.position();
        (*self.started.get(k)?).then(|| &self.square_avg[k][..])
    }

    /// One step on `path`: [`Optimizer::step_with_stats`] with the loops chosen explicitly.
    fn update(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats, path: Path) {
        if self.sizes.prepare("RmsProp", params) {
            let zeros = |lens: &[usize]| lens.iter().map(|&n| vec![T::ZERO; n]).collect();
            self.square_avg = zeros(&self.sizes.lens);
            if self.momentum != 0.0 {
                self.buffers = zeros(&self.sizes.lens);
            }
            self.started = vec![false; self.square_avg.len()];
            self.partials = vec![ChunkStats::default(); self.sizes.chunks()];
        }
        stats.reset(params.len());
        let c = RmsStep {
            lr: self.lr,
            alpha: self.alpha,
            eps: self.eps,
            mu: self.momentum,
            wd: self.weight_decay,
        };
        let momentum = self.momentum != 0.0;
        // A step that panicked may have left parts behind.
        self.plain.0.clear();
        self.with_momentum.0.clear();
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let grads = grad.as_ref().map(|g| g.data());
            if grads.is_some() {
                self.started[k] = true;
            }
            let avg = &mut self.square_avg[k][..];
            let buffer: &mut [T] = if momentum {
                &mut self.buffers[k][..]
            } else {
                &mut []
            };
            match (path, grads) {
                // No gradient since `zero_grad`: skipped, as PyTorch skips `grad is None`.
                (Path::Reference, None) => stats.push(id, 0.0, true),
                (Path::Reference, Some(g)) => {
                    let (sum, finite) = rms_reference(&c, value.data_mut(), avg, buffer, g);
                    stats.push(id, sum, finite);
                }
                (Path::Chunked(_), g) if momentum => {
                    let muts = match g {
                        Some(_) => [value.data_mut(), avg, buffer],
                        None => [&mut [][..], &mut [], &mut []],
                    };
                    self.with_momentum.push(id, muts, g, ());
                }
                (Path::Chunked(_), g) => {
                    let muts = match g {
                        Some(_) => [value.data_mut(), avg],
                        None => [&mut [][..], &mut []],
                    };
                    self.plain.push(id, muts, g, ());
                }
            }
        }
        let Path::Chunked(threads) = path else {
            return;
        };
        let parts = &mut self.partials;
        let decay = c.wd != 0.0;
        if momentum {
            run_parts(
                threads,
                &mut self.with_momentum,
                parts,
                stats,
                &|_, [p, v, b], g| {
                    if decay {
                        rms_chunk::<T, true, true>(&c, p, v, b, g)
                    } else {
                        rms_chunk::<T, false, true>(&c, p, v, b, g)
                    }
                },
            );
        } else {
            run_parts(threads, &mut self.plain, parts, stats, &|_, [p, v], g| {
                if decay {
                    rms_chunk::<T, true, false>(&c, p, v, &mut [], g)
                } else {
                    rms_chunk::<T, false, false>(&c, p, v, &mut [], g)
                }
            });
        }
    }
}

/// The settings of one RMSprop step.
#[derive(Clone, Copy, Debug)]
struct RmsStep {
    /// The learning rate.
    lr: f64,
    /// The decay rate of the squared-gradient average.
    alpha: f64,
    /// Added to the square root of the average.
    eps: f64,
    /// The momentum factor.
    mu: f64,
    /// The weight decay.
    wd: f64,
}

/// One element of RMSprop: from the value `p`, the gradient `g` (weight decay already
/// added) and the old average `v` (and buffer `b` with momentum), the new value, average
/// and buffer.
#[inline(always)]
fn rms_element(c: &RmsStep, momentum: bool, p: f64, g: f64, v: f64, b: f64) -> (f64, f64, f64) {
    let v = c.alpha * v + (1.0 - c.alpha) * g * g;
    let ratio = g / (v.sqrt() + c.eps);
    if momentum {
        let b = c.mu * b + ratio;
        (p - c.lr * b, v, b)
    } else {
        (p - c.lr * ratio, v, b)
    }
}

/// The plain RMSprop loop over one parameter, element by element on the calling thread:
/// the reference for [`rms_chunk`]. `buffer` is the momentum buffer (empty without
/// momentum). Returns the gradient's sum of squares and whether it was all finite.
fn rms_reference<T: Real>(
    c: &RmsStep,
    values: &mut [T],
    avg: &mut [T],
    buffer: &mut [T],
    grads: &[T],
) -> (f64, bool) {
    let momentum = c.mu != 0.0;
    let (mut sum, mut finite) = (0.0f64, true);
    for i in 0..values.len() {
        let raw = grads[i].to_f64();
        sum += raw * raw;
        finite &= raw.is_finite();
        let p = values[i].to_f64();
        let mut g = raw;
        if c.wd != 0.0 {
            g += c.wd * p;
        }
        let b = if momentum { buffer[i].to_f64() } else { 0.0 };
        let (p, v, b) = rms_element(c, momentum, p, g, avg[i].to_f64(), b);
        avg[i] = store_state(v);
        if momentum {
            buffer[i] = store_state(b);
        }
        values[i] = T::from_f64(p);
    }
    (sum, finite)
}

/// One chunk of RMSprop, with weight decay when `WD` and the momentum buffer `buffer` when
/// `MOMENTUM` (empty otherwise): the arithmetic of [`rms_reference`] per element.
fn rms_chunk<T: Real, const WD: bool, const MOMENTUM: bool>(
    c: &RmsStep,
    values: &mut [T],
    avg: &mut [T],
    buffer: &mut [T],
    grads: &[T],
) -> ChunkStats {
    let element = |p: &mut T, v: &mut T, b: Option<&mut T>, raw: f64| {
        let x = p.to_f64();
        let mut g = raw;
        if WD {
            g += c.wd * x;
        }
        let old_b = b.as_ref().map_or(0.0, |b| b.to_f64());
        let (x, nv, nb) = rms_element(c, MOMENTUM, x, g, v.to_f64(), old_b);
        *v = store_state(nv);
        if let Some(b) = b {
            *b = store_state(nb);
        }
        *p = T::from_f64(x);
    };
    if MOMENTUM {
        sweep([values, avg, buffer], grads, |[p, v, b], raw| {
            element(p, v, Some(b), raw)
        })
    } else {
        sweep([values, avg], grads, |[p, v], raw| element(p, v, None, raw))
    }
}

impl<T: Real> Optimizer<T> for RmsProp<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.update(params, stats, Path::current());
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.stats);
        self.step_with_stats(params, &mut stats);
        self.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.stats
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Adagrad (Duchi, Hazan and Singer, 2011) with PyTorch's `torch.optim.Adagrad` semantics:
/// a running sum of squared gradients, `s ← s + g²`, divides each step,
/// `p ← p − lr_t·g / (√s + eps)`, with weight decay added to the gradient as `wd·p` and the
/// rate decayed per update, `lr_t = lr / (1 + (t − 1)·lr_decay)` for a parameter's update
/// `t` (from 1; a step that skips the parameter does not count). The sum starts at
/// `initial_accumulator`. Defaults as PyTorch's: no rate decay, no weight decay, a sum
/// starting at 0, eps 1e-10.
///
/// # Examples
///
/// ```
/// use cavitation_core::{ParamStore, Tensor};
/// use cavitation_nn::{Adagrad, Optimizer};
/// let mut store = ParamStore::new();
/// let p = store.add(Tensor::full(&[1], 1.0f32));
/// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
/// Adagrad::new(0.1).step(&mut store);
/// // s = 4, so the first step is 0.1·2 / 2 = 0.1.
/// assert!((store.value(p).data()[0] - 0.9).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct Adagrad<T: Real = f32> {
    /// The learning rate.
    lr: f64,
    /// The rate's decay per update.
    lr_decay: f64,
    /// The weight decay added to the gradient.
    weight_decay: f64,
    /// The value every sum starts at.
    initial_accumulator: f64,
    /// Added to the square root of the sum, outside the root.
    eps: f64,
    /// The updates each parameter has had so far, for its rate decay.
    steps: Vec<u64>,
    /// The sizes seen on the first step.
    sizes: Sizes,
    /// The sum of squared gradients of each parameter.
    sums: Vec<Vec<T>>,
    /// The statistics of the last `step`.
    stats: GradStats,
    /// One slot per chunk of a step, for the chunks' statistics.
    partials: Vec<ChunkStats>,
    /// The parameters of a chunked step, each with its decayed rate (empty between steps).
    parts: Parts<T, f64, 2>,
}

impl<T: Real> Adagrad<T> {
    /// Adagrad with learning rate `lr`, no rate decay, no weight decay, sums starting at 0
    /// and eps 1e-10.
    ///
    /// ```
    /// let opt: cavitation_nn::Adagrad = cavitation_nn::Adagrad::new(0.01);
    /// # let _ = opt;
    /// ```
    pub fn new(lr: f64) -> Adagrad<T> {
        Adagrad {
            lr,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator: 0.0,
            eps: 1e-10,
            steps: Vec::new(),
            sizes: Sizes::default(),
            sums: Vec::new(),
            stats: GradStats::new(),
            partials: Vec::new(),
            parts: Parts::new(),
        }
    }

    /// Sets the rate's decay per update.
    ///
    /// ```
    /// let opt: cavitation_nn::Adagrad = cavitation_nn::Adagrad::new(0.01).with_lr_decay(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_lr_decay(mut self, lr_decay: f64) -> Adagrad<T> {
        self.lr_decay = lr_decay;
        self
    }

    /// Sets the weight decay, added to the gradient as `wd·p`.
    ///
    /// ```
    /// let opt: cavitation_nn::Adagrad = cavitation_nn::Adagrad::new(0.01).with_weight_decay(1e-4);
    /// # let _ = opt;
    /// ```
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Adagrad<T> {
        self.weight_decay = weight_decay;
        self
    }

    /// Sets the value every sum starts at (PyTorch's `initial_accumulator_value`).
    ///
    /// # Panics
    ///
    /// Panics if `value` is negative or not finite.
    ///
    /// ```
    /// let opt: cavitation_nn::Adagrad = cavitation_nn::Adagrad::new(0.01).with_initial_accumulator(0.1);
    /// # let _ = opt;
    /// ```
    pub fn with_initial_accumulator(mut self, value: f64) -> Adagrad<T> {
        assert!(
            value.is_finite() && value >= 0.0,
            "Adagrad initial_accumulator must be finite and at least 0, got {value}"
        );
        self.initial_accumulator = value;
        self
    }

    /// Sets the constant added to the square root of the sum.
    ///
    /// ```
    /// let opt: cavitation_nn::Adagrad = cavitation_nn::Adagrad::new(0.01).with_eps(1e-8);
    /// # let _ = opt;
    /// ```
    pub fn with_eps(mut self, eps: f64) -> Adagrad<T> {
        self.eps = eps;
        self
    }

    /// The sum of squared gradients of parameter `id`, or `None` until it has been updated
    /// once. The ids are those of the store the optimiser steps.
    ///
    /// ```
    /// use cavitation_core::{ParamStore, Tensor};
    /// use cavitation_nn::{Adagrad, Optimizer};
    /// let mut store = ParamStore::new();
    /// let p = store.add(Tensor::full(&[1], 1.0f32));
    /// let mut opt = Adagrad::new(0.1);
    /// assert!(opt.sum(p).is_none());
    /// store.grads.accumulate(p).0.data_mut()[0] = 2.0;
    /// opt.step(&mut store);
    /// assert_eq!(opt.sum(p), Some(&[4.0f32][..]));
    /// ```
    pub fn sum(&self, id: ParamId) -> Option<&[T]> {
        let k = id.position();
        (*self.steps.get(k)? > 0).then(|| &self.sums[k][..])
    }

    /// One step on `path`: [`Optimizer::step_with_stats`] with the loops chosen explicitly.
    fn update(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats, path: Path) {
        if self.sizes.prepare("Adagrad", params) {
            let start = store_state::<T>(self.initial_accumulator);
            self.sums = self.sizes.lens.iter().map(|&n| vec![start; n]).collect();
            self.steps = vec![0; self.sums.len()];
            self.partials = vec![ChunkStats::default(); self.sizes.chunks()];
        }
        stats.reset(params.len());
        let (wd, eps) = (self.weight_decay, self.eps);
        // A step that panicked may have left parts behind.
        self.parts.0.clear();
        for (k, (id, value, grad)) in params.iter_mut().enumerate() {
            let Some(grad) = grad else {
                // No gradient since `zero_grad`: skipped, as PyTorch skips `grad is None`
                // (in a chunked step, a part with no chunks).
                match path {
                    Path::Reference => stats.push(id, 0.0, true),
                    Path::Chunked(_) => self.parts.push(id, [&mut [], &mut []], None, 0.0),
                }
                continue;
            };
            self.steps[k] += 1;
            let rate = self.lr / (1.0 + (self.steps[k] - 1) as f64 * self.lr_decay);
            let (values, sums) = (value.data_mut(), &mut self.sums[k][..]);
            match path {
                Path::Reference => {
                    let (sum, finite) = adagrad_reference(rate, wd, eps, values, sums, grad.data());
                    stats.push(id, sum, finite);
                }
                Path::Chunked(_) => self.parts.push(id, [values, sums], Some(grad.data()), rate),
            }
        }
        let Path::Chunked(threads) = path else {
            return;
        };
        let (list, parts) = (&mut self.parts, &mut self.partials);
        if wd != 0.0 {
            run_parts(threads, list, parts, stats, &|&rate, [p, s], g| {
                adagrad_chunk::<T, true>(rate, wd, eps, p, s, g)
            });
        } else {
            run_parts(threads, list, parts, stats, &|&rate, [p, s], g| {
                adagrad_chunk::<T, false>(rate, wd, eps, p, s, g)
            });
        }
    }
}

/// One element of Adagrad: from the value `p`, the gradient `g` (weight decay already
/// added) and the old sum `s`, the new value and sum.
#[inline(always)]
fn adagrad_element(rate: f64, eps: f64, p: f64, g: f64, s: f64) -> (f64, f64) {
    let s = s + g * g;
    (p - rate * g / (s.sqrt() + eps), s)
}

/// The plain Adagrad loop over one parameter, element by element on the calling thread,
/// at the parameter's decayed rate `rate`: the reference for [`adagrad_chunk`]. Returns the
/// gradient's sum of squares and whether it was all finite.
fn adagrad_reference<T: Real>(
    rate: f64,
    wd: f64,
    eps: f64,
    values: &mut [T],
    sums: &mut [T],
    grads: &[T],
) -> (f64, bool) {
    let (mut sum, mut finite) = (0.0f64, true);
    for i in 0..values.len() {
        let raw = grads[i].to_f64();
        sum += raw * raw;
        finite &= raw.is_finite();
        let p = values[i].to_f64();
        let mut g = raw;
        if wd != 0.0 {
            g += wd * p;
        }
        let (p, s) = adagrad_element(rate, eps, p, g, sums[i].to_f64());
        sums[i] = store_state(s);
        values[i] = T::from_f64(p);
    }
    (sum, finite)
}

/// One chunk of Adagrad, with weight decay when `WD`: the arithmetic of
/// [`adagrad_reference`] per element.
fn adagrad_chunk<T: Real, const WD: bool>(
    rate: f64,
    wd: f64,
    eps: f64,
    values: &mut [T],
    sums: &mut [T],
    grads: &[T],
) -> ChunkStats {
    sweep([values, sums], grads, |[value, s], raw| {
        let p = value.to_f64();
        let mut g = raw;
        if WD {
            g += wd * p;
        }
        let (p, ns) = adagrad_element(rate, eps, p, g, s.to_f64());
        *s = store_state(ns);
        *value = T::from_f64(p);
    })
}

impl<T: Real> Optimizer<T> for Adagrad<T> {
    fn step_with_stats(&mut self, params: &mut ParamStore<T>, stats: &mut GradStats) {
        self.update(params, stats, Path::current());
    }

    fn step(&mut self, params: &mut ParamStore<T>) {
        let mut stats = std::mem::take(&mut self.stats);
        self.step_with_stats(params, &mut stats);
        self.stats = stats;
    }

    fn last_stats(&self) -> &GradStats {
        &self.stats
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// Tests for the optimisers.
#[cfg(test)]
mod tests {
    use super::*;
    use cavitation_core::Tensor;

    /// Sets the gradient of `p` to `values`.
    fn set_grad(store: &mut ParamStore, p: ParamId, values: &[f32]) {
        store
            .grads
            .accumulate(p)
            .0
            .data_mut()
            .copy_from_slice(values);
    }

    /// p = 1, g = 0.5, lr 0.1 gives 0.95.
    #[test]
    fn sgd_by_hand() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 1.0f32));
        set_grad(&mut store, p, &[0.5]);
        Sgd::new(0.1).step(&mut store);
        assert!((store.value(p).data()[0] - 0.95).abs() < 1e-7);
    }

    /// The first Adam step moves every parameter by lr in the direction of its gradient.
    #[test]
    fn adam_first_step_moves_by_lr() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::from_vec(vec![1.0f32, -2.0, 0.5, 3.0], &[4]).unwrap());
        let grads = [0.3f32, -7.0, 1e-3, -200.0];
        set_grad(&mut store, p, &grads);
        let before = store.value(p).data().to_vec();
        Adam::new(0.01).step(&mut store);
        for i in 0..4 {
            let moved = store.value(p).data()[i] - before[i];
            assert!(
                (moved - (-0.01 * grads[i].signum())).abs() < 1e-6,
                "element {i}"
            );
        }
    }

    /// The statistics equal a direct computation, report the first non-finite parameter and
    /// treat a stale gradient as zeros.
    #[test]
    fn grad_stats_match_direct_computation() {
        let mut store = ParamStore::new();
        let a = store.add(Tensor::zeros(&[3]));
        let b = store.add(Tensor::zeros(&[2]));
        let c = store.add(Tensor::zeros(&[2]));
        let d = store.add(Tensor::zeros(&[1]));
        set_grad(&mut store, a, &[1.0, -2.0, 2.0]);
        set_grad(&mut store, c, &[f32::NAN, 1.0]);
        set_grad(&mut store, d, &[f32::NEG_INFINITY]);
        let mut stats = GradStats::new();
        Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
        assert_eq!(stats.norm(a), 3.0);
        assert_eq!(stats.norm(b), 0.0);
        assert!(stats.norm(c).is_nan());
        assert!(!stats.all_finite());
        assert_eq!(stats.first_non_finite(), Some(c));
        store.zero_grad();
        Sgd::new(0.1).step_with_stats(&mut store, &mut stats);
        assert!(stats.all_finite());
        assert_eq!(stats.first_non_finite(), None);
        assert_eq!(stats.norm(a), 0.0);
    }

    /// Changing the number of parameters after the first step panics with a clear message.
    #[test]
    #[should_panic(expected = "the store has 2 parameters but the first step saw 1")]
    fn adding_a_parameter_after_the_first_step_panics() {
        let mut store = ParamStore::new();
        store.add(Tensor::zeros(&[2]));
        let mut opt = Adam::new(0.01);
        opt.step(&mut store);
        store.add(Tensor::zeros(&[2]));
        opt.step(&mut store);
    }

    /// Each optimiser returns the rate it was given, and its next step uses it: SGD moves by
    /// `lr·g`, Adam's first step moves by `lr`, and AdamW's decoupled decay shrinks by
    /// `lr·wd` (PyTorch's `p ← p·(1 − lr·wd)` with the group's current rate).
    #[test]
    fn set_learning_rate_round_trips_and_applies() {
        let one = || {
            let mut store = ParamStore::new();
            let p = store.add(Tensor::full(&[1], 1.0f32));
            (store, p)
        };

        let (mut store, p) = one();
        let mut sgd = Sgd::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut sgd, 0.25);
        assert_eq!(Optimizer::<f32>::learning_rate(&sgd), 0.25);
        set_grad(&mut store, p, &[1.0]);
        sgd.step(&mut store);
        assert_eq!(store.value(p).data(), &[0.75]);

        let (mut store, p) = one();
        let mut adam = Adam::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut adam, 0.03);
        assert_eq!(Optimizer::<f32>::learning_rate(&adam), 0.03);
        set_grad(&mut store, p, &[2.0]);
        adam.step(&mut store);
        assert!((store.value(p).data()[0] - 0.97).abs() < 1e-6);

        let (mut store, p) = one();
        let mut adamw = AdamW::new(0.1);
        Optimizer::<f32>::set_learning_rate(&mut adamw, 0.2);
        assert_eq!(Optimizer::<f32>::learning_rate(&adamw), 0.2);
        // A zero gradient: only the decay acts, 1 − 0.2 · 0.01.
        set_grad(&mut store, p, &[0.0]);
        adamw.step(&mut store);
        assert!((store.value(p).data()[0] - 0.998).abs() < 1e-7);
    }

    /// Values for the chunked-update tests: a fixed pseudo-random sequence (mulberry32-style
    /// mixing) in about [-2, 2), with `specials` dropped in at fixed places.
    fn sequence(len: usize, seed: u32, specials: &[f32]) -> Vec<f32> {
        let mut state = seed;
        let mut out: Vec<f32> = (0..len)
            .map(|_| {
                state = state.wrapping_add(0x6d2b_79f5);
                let mut z = state;
                z = (z ^ (z >> 15)).wrapping_mul(z | 1);
                z ^= z.wrapping_add((z ^ (z >> 7)).wrapping_mul(z | 61));
                ((z ^ (z >> 14)) as f32 / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect();
        for (k, &v) in specials.iter().enumerate() {
            out[(k * 7919) % len] = v;
        }
        out
    }

    /// A store of four parameters around the chunk size (`2·CHUNK + 37`, `CHUNK`, `CHUNK - 1`
    /// and 5 elements), so chunked updates cross chunk boundaries and leave short tails.
    fn chunk_store() -> (ParamStore, Vec<ParamId>) {
        let mut store = ParamStore::new();
        let ids = [2 * CHUNK + 37, CHUNK, CHUNK - 1, 5]
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                store.add(Tensor::from_vec(sequence(n, 11 + k as u32, &[0.0, -0.0]), &[n]).unwrap())
            })
            .collect();
        (store, ids)
    }

    /// Gives every parameter a fresh gradient for step `step`; the largest one also gets
    /// -0, a subnormal and a huge value, and on step 3 the 5-element one gets a NaN and an
    /// infinity.
    fn chunk_grads(store: &mut ParamStore, ids: &[ParamId], step: u32) {
        for (k, &id) in ids.iter().enumerate() {
            let n = store.value(id).data().len();
            let specials: &[f32] = match (k, step) {
                (0, _) => &[-0.0, 1e-40, 3e30],
                (3, 3) => &[f32::NAN, f32::INFINITY],
                _ => &[],
            };
            let g = sequence(n, 100 * step + k as u32, specials);
            store.grads.accumulate(id).0.data_mut().copy_from_slice(&g);
        }
    }

    /// Every optimiser setting the chunked path specialises: SGD with and without momentum
    /// and weight decay, Adam with and without weight decay, AdamW.
    fn chunk_configs() -> Vec<(&'static str, MakeOptimizer)> {
        vec![
            ("sgd", Box::new(|| Either::Sgd(Sgd::new(0.1)))),
            (
                "sgd wd",
                Box::new(|| Either::Sgd(Sgd::new(0.1).with_weight_decay(0.01))),
            ),
            (
                "sgd momentum",
                Box::new(|| Either::Sgd(Sgd::new(0.1).with_momentum(0.9))),
            ),
            (
                "sgd momentum wd",
                Box::new(|| Either::Sgd(Sgd::new(0.1).with_momentum(0.9).with_weight_decay(0.01))),
            ),
            (
                "adam",
                Box::new(|| Either::Adam(AdamCore::new(0.01, 0.0), false)),
            ),
            (
                "adam wd",
                Box::new(|| Either::Adam(AdamCore::new(0.01, 0.1), false)),
            ),
            (
                "adamw",
                Box::new(|| Either::Adam(AdamCore::new(0.01, 0.01), true)),
            ),
            (
                "adamw no decay",
                Box::new(|| Either::Adam(AdamCore::new(0.01, 0.0), true)),
            ),
            ("rmsprop", Box::new(|| Either::Rms(RmsProp::new(0.01)))),
            (
                "rmsprop wd",
                Box::new(|| Either::Rms(RmsProp::new(0.01).with_weight_decay(0.01))),
            ),
            (
                "rmsprop momentum",
                Box::new(|| Either::Rms(RmsProp::new(0.01).with_momentum(0.9))),
            ),
            (
                "rmsprop momentum wd",
                Box::new(|| {
                    Either::Rms(
                        RmsProp::new(0.01)
                            .with_momentum(0.9)
                            .with_weight_decay(0.01),
                    )
                }),
            ),
            ("adagrad", Box::new(|| Either::Adagrad(Adagrad::new(0.1)))),
            (
                "adagrad decay wd start",
                Box::new(|| {
                    Either::Adagrad(
                        Adagrad::new(0.1)
                            .with_lr_decay(0.05)
                            .with_weight_decay(0.01)
                            .with_initial_accumulator(0.1),
                    )
                }),
            ),
        ]
    }

    /// Builds a fresh optimiser for one of the chunked-update tests.
    type MakeOptimizer = Box<dyn Fn() -> Either>;

    /// An SGD, an Adam core, an RMSprop or an Adagrad, so the tests can call any update with an explicit path and
    /// read its state.
    enum Either {
        /// SGD.
        Sgd(Sgd),
        /// Adam (false) or AdamW (true).
        Adam(AdamCore<f32>, bool),
        /// RMSprop.
        Rms(RmsProp),
        /// Adagrad.
        Adagrad(Adagrad),
    }

    impl Either {
        /// One step on `path`.
        fn step(&mut self, store: &mut ParamStore, stats: &mut GradStats, path: Path) {
            match self {
                Either::Sgd(o) => o.update(store, stats, path),
                Either::Adam(o, decoupled) => o.step("Adam", *decoupled, store, stats, path),
                Either::Rms(o) => o.update(store, stats, path),
                Either::Adagrad(o) => o.update(store, stats, path),
            }
        }

        /// The state buffers of parameter `k` (none for SGD without momentum), as bits.
        fn state_of(&self, k: usize) -> Vec<Vec<u32>> {
            let bits = |b: &Vec<f32>| -> Vec<u32> { b.iter().map(|x| x.to_bits()).collect() };
            match self {
                Either::Sgd(o) => o.buffers.get(k).map(bits).into_iter().collect(),
                Either::Adam(o, _) => vec![bits(&o.m[k]), bits(&o.v[k])],
                Either::Rms(o) => std::iter::once(bits(&o.square_avg[k]))
                    .chain(o.buffers.get(k).map(bits))
                    .collect(),
                Either::Adagrad(o) => vec![bits(&o.sums[k])],
            }
        }

        /// The optimiser's state buffers, as bits.
        fn state_bits(&self) -> Vec<Vec<u32>> {
            let bits = |v: &Vec<Vec<f32>>| -> Vec<Vec<u32>> {
                v.iter()
                    .map(|b| b.iter().map(|x| x.to_bits()).collect())
                    .collect()
            };
            match self {
                Either::Sgd(o) => bits(&o.buffers),
                Either::Adam(o, _) => [bits(&o.m), bits(&o.v)].concat(),
                Either::Rms(o) => [bits(&o.square_avg), bits(&o.buffers)].concat(),
                Either::Adagrad(o) => bits(&o.sums),
            }
        }
    }

    /// Every parameter's value, as bits.
    fn value_bits(store: &ParamStore, ids: &[ParamId]) -> Vec<Vec<u32>> {
        ids.iter()
            .map(|&id| store.value(id).data().iter().map(|x| x.to_bits()).collect())
            .collect()
    }

    /// The chunked update gives the same parameters and the same optimiser state, bit for
    /// bit, as the reference loop, for every setting, over 3 steps (the last with a NaN and
    /// an infinity in one gradient) and on 1, 2 and 4 threads. Its statistics differ from
    /// the reference only by the order of the sum: within 1e-12 relative. Runs alone in a
    /// child process, so the 2- and 4-thread runs really use the pool's workers.
    #[test]
    fn chunked_update_matches_the_reference_bit_for_bit() {
        let out = crate::test_child::run_alone(
            "optim::tests::chunked_update_matches_the_reference_child",
            &[],
        );
        crate::test_child::assert_passed(&out);
    }

    /// The body of `chunked_update_matches_the_reference_bit_for_bit`, run in a child
    /// process after checking that the pool runs tasks on its workers.
    #[test]
    #[ignore = "run in a child process by chunked_update_matches_the_reference_bit_for_bit"]
    fn chunked_update_matches_the_reference_child() {
        crate::test_child::assert_pool_runs_on_workers();
        for (name, make) in chunk_configs() {
            let (mut reference, ids) = chunk_store();
            let mut ref_opt = make();
            let mut ref_stats = GradStats::new();
            let mut runs: Vec<_> = [1, 2, 4]
                .into_iter()
                .map(|threads| {
                    let (store, ids) = chunk_store();
                    (threads, store, ids, make(), GradStats::new())
                })
                .collect();
            let position = |stats: &GradStats, ids: &[ParamId]| {
                stats
                    .first_non_finite()
                    .map(|id| ids.iter().position(|&i| i == id))
            };
            for step in 1..=3 {
                chunk_grads(&mut reference, &ids, step);
                ref_opt.step(&mut reference, &mut ref_stats, Path::Reference);
                for (threads, store, own, opt, stats) in &mut runs {
                    chunk_grads(store, own, step);
                    opt.step(store, stats, Path::Chunked(*threads));
                    let at = format!("{name}, step {step}, {threads} threads");
                    assert!(
                        value_bits(store, own) == value_bits(&reference, &ids),
                        "{at}: values"
                    );
                    assert!(opt.state_bits() == ref_opt.state_bits(), "{at}: state");
                    for (&id, &ref_id) in own.iter().zip(&ids) {
                        let (a, b) = (stats.norm(id), ref_stats.norm(ref_id));
                        assert!(
                            a == b
                                || (a - b).abs() <= 1e-12 * b.abs()
                                || (a.is_nan() && b.is_nan()),
                            "{at}: norm {a} vs reference {b}"
                        );
                    }
                    assert_eq!(position(stats, own), position(&ref_stats, &ids), "{at}");
                }
            }
        }
    }

    /// The chunked statistics are bit-identical on 1, 2 and 4 threads: each chunk sums in
    /// fixed partial sums and the chunks combine in order. Runs alone in a child process, so
    /// the 2- and 4-thread runs really use the pool's workers.
    #[test]
    fn chunked_statistics_do_not_depend_on_the_thread_count() {
        let out = crate::test_child::run_alone("optim::tests::chunked_statistics_child", &[]);
        crate::test_child::assert_passed(&out);
    }

    /// The body of `chunked_statistics_do_not_depend_on_the_thread_count`, run in a child
    /// process after checking that the pool runs tasks on its workers.
    #[test]
    #[ignore = "run in a child process by chunked_statistics_do_not_depend_on_the_thread_count"]
    fn chunked_statistics_child() {
        crate::test_child::assert_pool_runs_on_workers();
        for (name, make) in chunk_configs() {
            let norms: Vec<Vec<u64>> = [1, 2, 4]
                .into_iter()
                .map(|threads| {
                    let (mut store, ids) = chunk_store();
                    let mut opt = make();
                    let mut stats = GradStats::new();
                    let mut bits = Vec::new();
                    for step in 1..=3 {
                        chunk_grads(&mut store, &ids, step);
                        opt.step(&mut store, &mut stats, Path::Chunked(threads));
                        bits.extend(ids.iter().map(|&id| stats.norm(id).to_bits()));
                        bits.extend(
                            ids.iter()
                                .map(|&id| u64::from(stats.first_non_finite() == Some(id))),
                        );
                    }
                    bits
                })
                .collect();
            assert!(norms[0] == norms[1] && norms[0] == norms[2], "{name}");
        }
    }

    /// With parameters left out anywhere in the store (the first, two in the middle, the
    /// last, the first and a middle one), a chunked step on 1, 2 and 4 threads gives the
    /// reference loop's values and optimiser state bit for bit, and its statistics within
    /// 1e-12 relative, with a zero norm for every parameter left out. Runs alone in a child
    /// process, so the 2- and 4-thread runs really use the pool's workers.
    #[test]
    fn chunked_update_with_skipped_parameters_matches_the_reference() {
        let out = crate::test_child::run_alone("optim::tests::chunked_skips_child", &[]);
        crate::test_child::assert_passed(&out);
    }

    /// The body of `chunked_update_with_skipped_parameters_matches_the_reference`, run in a
    /// child process after checking that the pool runs tasks on its workers.
    #[test]
    #[ignore = "run in a child process by chunked_update_with_skipped_parameters_matches_the_reference"]
    fn chunked_skips_child() {
        crate::test_child::assert_pool_runs_on_workers();
        let skips: [&[usize]; 4] = [&[0], &[1, 2], &[3], &[0, 2]];
        // Gradients for every parameter of the store except those in `skip`.
        let grads = |store: &mut ParamStore, ids: &[ParamId], step: u32, skip: &[usize]| {
            store.zero_grad();
            for (k, &id) in ids.iter().enumerate().filter(|(k, _)| !skip.contains(k)) {
                let n = store.value(id).data().len();
                set_grad(store, id, &sequence(n, 100 * step + k as u32, &[]));
            }
        };
        for (name, make) in chunk_configs() {
            let (mut reference, ids) = chunk_store();
            let mut ref_opt = make();
            let mut ref_stats = GradStats::new();
            let mut runs: Vec<_> = [1, 2, 4]
                .into_iter()
                .map(|threads| {
                    let (store, ids) = chunk_store();
                    (threads, store, ids, make(), GradStats::new())
                })
                .collect();
            for (step, skip) in (1..).zip(skips) {
                grads(&mut reference, &ids, step, skip);
                ref_opt.step(&mut reference, &mut ref_stats, Path::Reference);
                for (threads, store, own, opt, stats) in &mut runs {
                    grads(store, own, step, skip);
                    opt.step(store, stats, Path::Chunked(*threads));
                    let at = format!("{name}, step {step}, {threads} threads");
                    assert!(
                        value_bits(store, own) == value_bits(&reference, &ids),
                        "{at}: values"
                    );
                    assert!(opt.state_bits() == ref_opt.state_bits(), "{at}: state");
                    for (k, (&id, &ref_id)) in own.iter().zip(&ids).enumerate() {
                        let (a, b) = (stats.norm(id), ref_stats.norm(ref_id));
                        assert!(
                            a == b || (a - b).abs() <= 1e-12 * b.abs(),
                            "{at}: norm {a} vs reference {b}"
                        );
                        assert!(!skip.contains(&k) || a == 0.0, "{at}: skipped norm {a}");
                    }
                }
            }
        }
    }

    /// A parameter that gets no gradient is skipped, as PyTorch skips `grad is None`, by
    /// every optimiser setting on both paths: over three steps where only `a` gets a gradient,
    /// `b`'s value and its optimiser state stay bit-identical, and the statistics count its
    /// gradient as zeros. When `b` then gets its first gradient it moves exactly as a fresh
    /// optimiser's first step would move it (its own step count, its momentum buffer started
    /// from that gradient), while `a` keeps matching the reference.
    #[test]
    fn parameters_without_a_gradient_are_skipped() {
        for (name, make) in chunk_configs() {
            for path in [Path::Reference, Path::Chunked(2)] {
                let at = format!("{name}, {path:?}");
                let mut store = ParamStore::new();
                let a = store.add(Tensor::from_vec(sequence(40, 1, &[]), &[40]).unwrap());
                let b = store.add(Tensor::from_vec(sequence(30, 2, &[]), &[30]).unwrap());
                let b_start = value_bits(&store, &[b]);
                let mut opt = make();
                let mut stats = GradStats::new();
                // The state `b` starts with: zeros, or Adagrad's initial accumulator.
                let mut b_state = None;
                for step in 1..=3 {
                    store.zero_grad();
                    set_grad(&mut store, a, &sequence(40, 10 + step, &[]));
                    opt.step(&mut store, &mut stats, path);
                    assert_eq!(
                        value_bits(&store, &[b]),
                        b_start,
                        "{at}, step {step}: value"
                    );
                    let state = opt.state_of(1);
                    let start = b_state.get_or_insert_with(|| state.clone());
                    assert!(
                        state == *start
                            && (matches!(opt, Either::Adagrad(_))
                                || state.iter().flatten().all(|&x| x == 0)),
                        "{at}, step {step}: b's state was written"
                    );
                    assert_eq!(stats.norm(b), 0.0, "{at}");
                }

                let gb = sequence(30, 20, &[]);
                store.zero_grad();
                set_grad(&mut store, a, &sequence(40, 14, &[]));
                set_grad(&mut store, b, &gb);
                opt.step(&mut store, &mut stats, path);
                let mut fresh = ParamStore::new();
                let only_b = fresh.add(Tensor::from_vec(sequence(30, 2, &[]), &[30]).unwrap());
                set_grad(&mut fresh, only_b, &gb);
                make().step(&mut fresh, &mut GradStats::new(), path);
                assert_eq!(
                    value_bits(&store, &[b]),
                    value_bits(&fresh, &[only_b]),
                    "{at}: b's first update"
                );
            }
        }
    }

    /// A parameter that already has momentum or Adam moments keeps them, and its value, while
    /// it has no gradient. For SGD with momentum 0.9, Adam and AdamW on both paths: both
    /// parameters get a gradient on step 1; only `a` gets one on steps 2 and 3, and after each
    /// of those `b`'s value bits and state bits equal what step 1 left. On step 4, when `b`
    /// has a gradient again, it moves exactly as the second update of an optimiser that only
    /// ever saw `b` (its own step count is 2, its state carried over from step 1).
    #[test]
    fn skipped_parameter_keeps_its_state() {
        for (name, make) in chunk_configs() {
            if !["sgd momentum", "adam", "adamw"].contains(&name) {
                continue;
            }
            for path in [Path::Reference, Path::Chunked(2)] {
                let at = format!("{name}, {path:?}");
                let (n_a, n_b) = (40, 30);
                let start_b = sequence(n_b, 2, &[]);
                let mut store = ParamStore::new();
                let a = store.add(Tensor::from_vec(sequence(n_a, 1, &[]), &[n_a]).unwrap());
                let b = store.add(Tensor::from_vec(start_b.clone(), &[n_b]).unwrap());
                let mut opt = make();
                let mut stats = GradStats::new();
                let (g1, g4) = (sequence(n_b, 20, &[]), sequence(n_b, 21, &[]));

                store.zero_grad();
                set_grad(&mut store, a, &sequence(n_a, 10, &[]));
                set_grad(&mut store, b, &g1);
                opt.step(&mut store, &mut stats, path);
                let value_after_1 = value_bits(&store, &[b]);
                let state_after_1 = opt.state_of(1);
                assert!(
                    state_after_1.iter().flatten().any(|&x| x != 0),
                    "{at}: step 1 left b's state at zero"
                );
                assert_ne!(
                    value_after_1,
                    vec![start_b.iter().map(|x| x.to_bits()).collect::<Vec<u32>>()],
                    "{at}: step 1 did not move b"
                );

                for step in 2..=3 {
                    store.zero_grad();
                    set_grad(&mut store, a, &sequence(n_a, 10 + step, &[]));
                    opt.step(&mut store, &mut stats, path);
                    assert_eq!(
                        value_bits(&store, &[b]),
                        value_after_1,
                        "{at}, step {step}: value"
                    );
                    assert_eq!(opt.state_of(1), state_after_1, "{at}, step {step}: state");
                }

                store.zero_grad();
                set_grad(&mut store, a, &sequence(n_a, 14, &[]));
                set_grad(&mut store, b, &g4);
                opt.step(&mut store, &mut stats, path);

                let mut twin = ParamStore::new();
                let only_b = twin.add(Tensor::from_vec(start_b, &[n_b]).unwrap());
                let mut twin_opt = make();
                let mut twin_stats = GradStats::new();
                for g in [&g1, &g4] {
                    twin.zero_grad();
                    set_grad(&mut twin, only_b, g);
                    twin_opt.step(&mut twin, &mut twin_stats, path);
                }
                assert_eq!(
                    value_bits(&store, &[b]),
                    value_bits(&twin, &[only_b]),
                    "{at}: b's update on step 4"
                );
                assert_eq!(
                    opt.state_of(1),
                    twin_opt.state_of(0),
                    "{at}: b's state on step 4"
                );
            }
        }
    }

    /// A `Box<dyn Optimizer>` is an optimiser.
    #[test]
    fn boxed_optimizer_steps() {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1], 1.0f32));
        set_grad(&mut store, p, &[1.0]);
        let mut opt: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(0.5));
        opt.set_learning_rate(0.25);
        opt.step(&mut store);
        assert_eq!(opt.learning_rate(), 0.25);
        assert_eq!(store.value(p).data(), &[0.75]);
    }

    /// Whether every value is 0 or at least `f32::MIN_POSITIVE` in magnitude (NaN passes).
    fn none_subnormal(values: &[f32]) -> bool {
        values
            .iter()
            .all(|x| *x == 0.0 || x.abs() >= f32::MIN_POSITIVE || x.is_nan())
    }

    /// A parameter of 1,000 elements that gets gradient 1e-3 on the first step and 0 on all
    /// later ones, so the optimiser's state decays through the subnormal range.
    fn decaying_store() -> (ParamStore, ParamId) {
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[1000], 1.0f32));
        (store, p)
    }

    /// Momentum's buffer never holds a subnormal value over 2,000 steps (it decays from 1e-3
    /// by a factor 0.9 a step, so it crosses `f32::MIN_POSITIVE` near step 800), on the
    /// chunked loops and on the reference loop, and ends at exactly 0.
    #[test]
    fn momentum_state_never_subnormal() {
        for path in [Path::Reference, Path::Chunked(1)] {
            let (mut store, p) = decaying_store();
            let mut opt = Sgd::new(0.01).with_momentum(0.9);
            let mut stats = GradStats::new();
            for step in 0..2000 {
                set_grad(
                    &mut store,
                    p,
                    &vec![if step == 0 { 1e-3 } else { 0.0 }; 1000],
                );
                opt.update(&mut store, &mut stats, path);
                let buf = opt.momentum(p).unwrap();
                assert!(none_subnormal(buf), "{path:?}: step {step}: {:e}", buf[0]);
            }
            assert!(
                opt.momentum(p).unwrap().iter().all(|&x| x == 0.0),
                "{path:?}"
            );
        }
    }

    /// Adam's moments never hold a subnormal value over 2,000 steps, on both paths: with
    /// the default betas the first moment crosses the bound near step 800, and with
    /// both betas 0.9 the second moment crosses it near step 400 too.
    #[test]
    fn adam_state_never_subnormal() {
        for path in [Path::Reference, Path::Chunked(1)] {
            for betas in [(0.9, 0.999), (0.9, 0.9)] {
                let (mut store, p) = decaying_store();
                let mut core = AdamCore::<f32>::new(1e-3, 0.0);
                core.set_betas(betas.0, betas.1);
                let mut stats = GradStats::new();
                for step in 0..2000 {
                    set_grad(
                        &mut store,
                        p,
                        &vec![if step == 0 { 1e-3 } else { 0.0 }; 1000],
                    );
                    core.step("Adam", false, &mut store, &mut stats, path);
                    let (m, v) = core.moments(p).unwrap();
                    assert!(
                        none_subnormal(m),
                        "{path:?} {betas:?}: step {step}: m {:e}",
                        m[0]
                    );
                    assert!(
                        none_subnormal(v),
                        "{path:?} {betas:?}: step {step}: v {:e}",
                        v[0]
                    );
                }
                let (m, _) = core.moments(p).unwrap();
                assert!(m.iter().all(|&x| x == 0.0), "{path:?} {betas:?}");
            }
        }
    }

    /// The accessors return `None` before a parameter's first update and the state after
    /// it: after two steps with gradient g, momentum 0.9 holds 1.9·g, and Adam and AdamW
    /// hold m = 0.19·g and v = 0.001999·g².
    #[test]
    fn state_accessors() {
        let g = 0.5f32;
        let mut store = ParamStore::new();
        let p = store.add(Tensor::full(&[3], 1.0f32));
        let mut sgd = Sgd::new(0.1).with_momentum(0.9);
        let mut plain = Sgd::<f32>::new(0.1);
        let mut adam = Adam::new(0.01);
        let mut adamw = AdamW::new(0.01);
        assert!(sgd.momentum(p).is_none() && adam.moments(p).is_none());
        assert!(adamw.moments(p).is_none());
        for _ in 0..2 {
            set_grad(&mut store, p, &[g; 3]);
            sgd.step(&mut store);
            plain.step(&mut store);
            adam.step(&mut store);
            adamw.step(&mut store);
        }
        assert!(plain.momentum(p).is_none());
        let close = |got: &[f32], want: f32| got.iter().all(|x| (x - want).abs() <= 1e-6 * want);
        assert!(close(sgd.momentum(p).unwrap(), 1.9 * g));
        for (m, v) in [adam.moments(p).unwrap(), adamw.moments(p).unwrap()] {
            assert!(close(m, 0.19 * g), "{m:?}");
            assert!(close(v, 0.001999 * g * g), "{v:?}");
        }
    }
}
