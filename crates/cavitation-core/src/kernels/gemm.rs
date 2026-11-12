//! The blocked, packed f32 GEMM: Goto-style blocking and packing around a SIMD microkernel,
//! split across the thread pool so that the result does not depend on the thread count.
//!
//! The product is cut into blocks of `nc` columns of `C` and `kc` steps of the sum. For each
//! block, `B`'s `kc` by `nc` slice is packed once into `nr`-column panels (the packing split
//! across threads), and then the threads share it read-only: each task takes a group of
//! `mr`-row panels of `A` and a group of `B` panels, packs its rows of `A`, and runs the
//! microkernel on every `mr` by `nr` tile of its part of `C`. Edges are handled by padding
//! the packed panels with zeros; only the valid part of a tile is stored.
//!
//! Every element of `C` is computed by one task, as the sum over the `kc` blocks in order,
//! each block summed in order inside the microkernel. The blocking depends only on the
//! shape, never on the thread count, and `k` is never split between threads, so any thread
//! count (and the single-thread path for small products) gives bit-identical results.
//!
//! # Examples
//!
//! The packed product agrees with its reference twin, [`reference::gemm`], here on a
//! product with `B` transposed:
//!
//! ```
//! use cavitation_core::kernels::gemm::gemm_f32;
//! use cavitation_core::kernels::{MatMut, MatRef, reference};
//! let (m, n, k) = (5, 7, 9);
//! let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
//! let bt: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.5 - 1.0).collect();
//! let a_view = MatRef::new(&a, m, k, k, 1).unwrap();
//! let b_view = MatRef::new(&bt, k, n, 1, k).unwrap(); // B stored as its transpose
//! let (mut fast, mut slow) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
//! gemm_f32(1.0, a_view, b_view, 0.0, MatMut::new(&mut fast, m, n, n, 1).unwrap());
//! reference::gemm(1.0, a_view, b_view, 0.0, MatMut::new(&mut slow, m, n, n, 1).unwrap());
//! for (f, s) in fast.iter().zip(&slow) {
//!     assert!((f - s).abs() <= 1e-5 * (1.0 + s.abs()));
//! }
//! ```

use std::cell::RefCell;

use super::{Level, MatMut, MatRef, pool, reference, supported};

/// A microkernel and the block sizes the driver uses with it.
///
/// # Safety
///
/// Implementations promise that [`Kernel::tile`] reads and writes only what its contract
/// allows, given that the caller upholds it.
pub(super) unsafe trait Kernel {
    /// Rows of `C` in one tile.
    const MR: usize;
    /// Columns of `C` in one tile.
    const NR: usize;
    /// The longest block of the sum over `k`.
    const KC: usize;
    /// The most rows of `A` one task packs.
    const MC: usize;
    /// The widest block of columns of `C`.
    const NC: usize;
    /// How many times each packed value of `A` is repeated (4 to store it broadcast to a
    /// four-wide vector, 1 to store it once).
    ///
    /// The driver starts every `A` panel on a 16-byte boundary only when `MR · A_REP` is a
    /// multiple of 4; otherwise a panel is aligned to f32 alone, and the kernel must not
    /// make aligned vector loads from it.
    const A_REP: usize;

    /// Packs rows `rows.0..rows.1` of `A`, columns `ks.0..ks.0 + ks.1`, into `dst` in the
    /// layout [`Kernel::tile`] reads: `MR`-row panels as [`pack_a`] describes, with
    /// `REP = A_REP`.
    fn pack_a(a: &MatRef<f32>, rows: (usize, usize), ks: (usize, usize), dst: &mut [f32]);

    /// Packs columns `cols.0..cols.0 + cols.1` of `B` (`cols.1 <= NR`), rows
    /// `ks.0..ks.0 + ks.1`, into `dst` as one panel in the layout [`Kernel::tile`] reads, as
    /// [`pack_b`] describes.
    fn pack_b(b: &MatRef<f32>, cols: (usize, usize), ks: (usize, usize), dst: &mut [f32]);

    /// Computes one tile: `C = alpha · Ã · B̃ + beta · C` for the valid `dest.rows` by
    /// `dest.cols` corner of an `MR` by `NR` tile, where `Ã` is the packed `A` panel (`kc`
    /// groups of `MR` values, each repeated `A_REP` times) and `B̃` the packed `B` panel (`kc`
    /// groups of `NR` values). Each element's sum runs over `p` in order from zero, each
    /// step a multiply and an add (fused into one rounding by kernels with fused
    /// multiply-add), and is then stored as `alpha · sum` when `beta` is 0 (C is not read) or
    /// `alpha · sum + beta · c` otherwise, the same way for edge tiles as for full ones.
    ///
    /// # Safety
    ///
    /// `a` must be valid for reading `MR · A_REP · kc` values and `b` for reading `NR · kc`
    /// values. `b` is aligned to 16 bytes; `a` is aligned to 16 bytes when `MR · A_REP` is a
    /// multiple of 4, and to 4 bytes otherwise. `dest.rows <= MR`, `dest.cols <= NR`, and for every
    /// `r < dest.rows` and `j < dest.cols` the element at `dest.ptr + r · dest.rs + j · dest.cs`
    /// must be valid for reading and writing, and not accessed by any other thread during
    /// the call. The CPU must support the kernel's instruction set.
    unsafe fn tile(kc: usize, a: *const f32, b: *const f32, alpha: f32, beta: f32, dest: &Dest);
}

/// Where a microkernel stores its tile: the tile's top-left element of `C` and the part of
/// the tile that lies inside `C`.
pub(super) struct Dest {
    /// The tile's top-left element.
    pub(super) ptr: *mut f32,
    /// Distance between vertically adjacent elements of `C`.
    pub(super) rs: usize,
    /// Distance between horizontally adjacent elements of `C`.
    pub(super) cs: usize,
    /// The rows of the tile inside `C`, at most `MR`.
    pub(super) rows: usize,
    /// The columns of the tile inside `C`, at most `NR`.
    pub(super) cols: usize,
}

/// Products of fewer floating-point operations than this (`2 · m · n · k`) run on the
/// calling thread alone, with the same blocking: about the size below which two thread-pool
/// round trips per block, and the workers kept spinning between them, cost more than
/// splitting the work saves. 2¹⁸ keeps a 64-row product with 32 on the other two sides
/// (2¹⁷) on one thread; at 2¹⁷ such a network's training used 4 times the CPU on 4 threads
/// for no gain in wall time. Measured on a 4-core Intel Celeron N5105 with
/// `cavitation-bench small`, `digits` and `mnist` (batch 128 and 32) at 2¹⁷, 2¹⁸, 2¹⁹ and
/// 2²⁰: 2¹⁸ left `digits` and `mnist` unchanged, 2¹⁹ and 2²⁰ made `digits` slower.
const SINGLE_THREAD_FLOPS: usize = 1 << 18;

/// Compute tasks per thread the driver aims for. Threads take tasks in order as they finish,
/// so more tasks would let fast cores take over work from slow ones, but each task packs
/// its rows of `A` again and the products get narrower. One was chosen with
/// `cavitation-bench gemm` and re-checked by timing whole training steps (see the SSE
/// kernel's notes in `x86.rs`), then again against 2 and 4 with `cavitation-bench all`
/// (medians of 3 interleaved runs, measured on an Intel Celeron N5105, 4 cores): 2 and 4
/// cost MNIST at batch 128 1.8% and 9.1% and digits training 12.1% and 28.5%; on an
/// Apple M4 at its default 4 threads, 2 gave no gain (0.96 against 0.97 s per MNIST epoch)
/// and cost digits 16%.
const TASKS_PER_THREAD: usize = 1;

/// General matrix multiply in f32, `C = alpha · A · B + beta · C`, with the kernel for the
/// current [`level`](super::level) on [`threads`](super::threads) threads.
///
/// Results match [`reference::gemm`] within the rounding of a different summation order
/// (the sum over `k` runs in blocks), are bit-identical for any thread count, and are the
/// same from run to run at the same level. When `beta` is zero the old contents of `C` are
/// never read, so NaN in `C` does not leak into the result.
///
/// Bit-identical results across thread count assume the default floating-point mode. The
/// pool's worker threads always run in it; a caller that turns on flush-to-zero or
/// denormals-are-zero (the x86 MXCSR flags) on its own thread computes its share of the
/// tiles in that mode, so results that involve subnormal values can then depend on the
/// thread count.
///
/// # Panics
///
/// Panics, naming the shapes, if `a.cols() != b.rows()` or `C` is not `a.rows()` by
/// `b.cols()`; and on first use, as [`level`](super::level) and [`threads`](super::threads)
/// do, if `CAVITATION_KERNEL` or `CAVITATION_THREADS` is invalid.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::gemm::gemm_f32;
/// use cavitation_core::kernels::{MatMut, MatRef};
/// let a = [1.0f32, 2.0, 3.0, 4.0];
/// let b = [5.0f32, 6.0, 7.0, 8.0];
/// let mut c = [0.0f32; 4];
/// gemm_f32(
///     1.0,
///     MatRef::new(&a, 2, 2, 2, 1).unwrap(),
///     MatRef::new(&b, 2, 2, 2, 1).unwrap(),
///     0.0,
///     MatMut::new(&mut c, 2, 2, 2, 1).unwrap(),
/// );
/// assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn gemm_f32(alpha: f32, a: MatRef<f32>, b: MatRef<f32>, beta: f32, c: MatMut<f32>) {
    gemm_f32_with(super::level(), super::threads(), alpha, a, b, beta, c);
}

/// [`gemm_f32`] with the kernel level and the thread count given explicitly, instead of
/// the process-wide settings.
///
/// [`Level::Sse42`] runs the packed SSE4.2 kernel (6 by 8 tiles), [`Level::Avx2Fma`] the packed
/// AVX2 kernel (6 by 16 tiles, fused multiply-add) and [`Level::Neon`] the packed NEON kernel (8 by
/// 12 tiles, fused multiply-add). [`Level::Accelerate`] (Apple Silicon) calls Accelerate's
/// `cblas_sgemm` once, on the calling thread, when every view is row-major or column-major with
/// rows (or columns) that do not overlap, padded or not, and fits in `i32` indices; any other
/// layout, or `alpha = 0`, runs the NEON kernel instead. [`Level::Reference`] runs
/// [`reference::gemm`] on one thread. Products too small to be worth splitting run on one thread;
/// [`threads_used`] tells how many a call uses. Each level rounds differently (fused multiply-add
/// rounds once per step, and each kernel blocks the sum differently), so results are bit-identical
/// across thread counts within one level, not across levels.
///
/// # Panics
///
/// Panics if `threads` is 0, if this CPU cannot run `level` (the message names the level
/// and the detected features), or, naming the shapes, if `a.cols() != b.rows()` or `C` is
/// not `a.rows()` by `b.cols()`. Nothing is computed before these checks.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::gemm::gemm_f32_with;
/// use cavitation_core::kernels::{detected, MatMut, MatRef};
/// let a = [1.0f32, 2.0];
/// let b = [3.0f32, 4.0];
/// let mut c = [10.0f32];
/// gemm_f32_with(
///     detected(),
///     2,
///     1.0,
///     MatRef::new(&a, 1, 2, 2, 1).unwrap(),
///     MatRef::new(&b, 2, 1, 1, 1).unwrap(),
///     0.5,
///     MatMut::new(&mut c, 1, 1, 1, 1).unwrap(),
/// );
/// assert_eq!(c, [16.0]);
/// ```
pub fn gemm_f32_with(
    level: Level,
    threads: usize,
    alpha: f32,
    a: MatRef<f32>,
    b: MatRef<f32>,
    beta: f32,
    c: MatMut<f32>,
) {
    assert!(
        threads >= 1,
        "gemm_f32_with: threads must be at least 1, got 0"
    );
    assert!(
        supported(level),
        "gemm_f32_with: {}",
        super::cannot_run(level)
    );
    reference::check_gemm_shapes(&a, &b, &c);
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    if m == 0 || n == 0 || k == 0 {
        // Nothing to multiply: C is untouched (m or n is 0) or becomes beta · C (k is 0).
        return reference::gemm(alpha, a, b, beta, c);
    }
    // Each arm runs a kernel only after checking that this CPU supports it.
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports AVX2 and FMA (the guard).
        Level::Avx2Fma if supported(Level::Avx2Fma) => unsafe {
            dispatch::<super::x86::Avx2>(threads, alpha, a, b, beta, c)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: this CPU supports SSE4.2 (the guard).
        Level::Sse42 if supported(Level::Sse42) => unsafe {
            dispatch::<super::x86::Sse>(threads, alpha, a, b, beta, c)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: this CPU supports NEON (the guard).
        Level::Neon if supported(Level::Neon) => unsafe {
            dispatch::<super::aarch64::Neon>(threads, alpha, a, b, beta, c)
        },
        #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
        Level::Accelerate if supported(Level::Accelerate) => {
            // One call into Accelerate on this thread; a layout BLAS cannot express runs the
            // NEON kernel instead. Which one runs depends only on the layout, never on
            // `threads`, so the result does not depend on the thread count either way.
            if let Err(c) = super::accelerate::gemm(alpha, a, b, beta, c) {
                // SAFETY: this CPU supports NEON: `supported(Level::Accelerate)` (the guard)
                // checks for it.
                unsafe { dispatch::<super::aarch64::Neon>(threads, alpha, a, b, beta, c) }
            }
        }
        _ => reference::gemm(alpha, a, b, beta, c),
    }
}

/// The packed product with kernel `K`, swapping to `Cᵀ = Bᵀ · Aᵀ` when [`swaps`] says so.
///
/// # Safety
///
/// As [`packed`]: the CPU must support `K`'s instruction set.
unsafe fn dispatch<K: Kernel>(
    threads: usize,
    alpha: f32,
    a: MatRef<f32>,
    b: MatRef<f32>,
    beta: f32,
    c: MatMut<f32>,
) {
    let (m, n) = (a.rows(), b.cols());
    if swaps(m, n) {
        // Cᵀ = Bᵀ · Aᵀ: the same sums of the same products in the same order (a product of
        // two floats does not depend on their order), so the same bits.
        // SAFETY: the caller guarantees the CPU supports `K`.
        unsafe {
            packed::<K>(
                threads,
                alpha,
                b.transpose(),
                a.transpose(),
                beta,
                c.transpose(),
            );
        }
    } else {
        // SAFETY: the caller guarantees the CPU supports `K`.
        unsafe { packed::<K>(threads, alpha, a, b, beta, c) }
    }
}

/// The number of threads [`gemm_f32_with`] uses for an `m` by `k` times `k` by `n` product
/// at `level` when given `threads`: 1 for the reference loops, for Accelerate's single call
/// (a layout it cannot express runs on `threads_used(Level::Neon, ..)` threads instead), for
/// empty products and for products too small to be worth splitting, else the most threads
/// any step of the product runs on (never more than its number of tasks, nor more than the
/// pool's maximum of 256).
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::gemm::threads_used;
/// assert_eq!(threads_used(Level::Reference, 4, 128, 1024, 784), 1);
/// assert_eq!(threads_used(Level::Sse42, 4, 2, 2, 2), 1);
/// ```
pub fn threads_used(level: Level, threads: usize, m: usize, n: usize, k: usize) -> usize {
    if m == 0 || n == 0 || k == 0 {
        return 1;
    }
    match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2Fma if supported(Level::Avx2Fma) => {
            plan_threads::<super::x86::Avx2>(threads, m, n, k)
        }
        #[cfg(target_arch = "x86_64")]
        Level::Sse42 if supported(Level::Sse42) => {
            plan_threads::<super::x86::Sse>(threads, m, n, k)
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon if supported(Level::Neon) => {
            plan_threads::<super::aarch64::Neon>(threads, m, n, k)
        }
        // The reference loops, and Accelerate's one call (any threads it starts itself are
        // its own).
        _ => 1,
    }
}

/// [`threads_used`] for kernel `K`, with `m`, `n` and `k` all non-zero: the threads
/// [`dispatch`] uses for this shape.
fn plan_threads<K: Kernel>(threads: usize, m: usize, n: usize, k: usize) -> usize {
    let (m, n) = if swaps(m, n) { (n, m) } else { (m, n) };
    Plan::new::<K>(m, n, k, threads).threads_used::<K>(n)
}

/// Whether the packed driver computes `Cᵀ = Bᵀ · Aᵀ` instead: when `C` is much taller than
/// wide. Every task packs its own rows of `A` (the SSE4.2 kernel stores each value four times
/// over), so packing costs most when `A` is the large operand and every packed row serves
/// only a few columns.
fn swaps(m: usize, n: usize) -> bool {
    n.saturating_mul(4) <= m
}

/// The thread count for the packed driver: 1 below [`SINGLE_THREAD_FLOPS`], else `threads`.
fn threads_for(m: usize, n: usize, k: usize, threads: usize) -> usize {
    let flops = m.saturating_mul(n).saturating_mul(k).saturating_mul(2);
    if flops < SINGLE_THREAD_FLOPS {
        1
    } else {
        threads
    }
}

/// The threads one `pool::run` of `tasks` tasks uses when given `threads`: the pool never
/// runs more threads than tasks, nor more than its maximum.
fn launch(threads: usize, tasks: usize) -> usize {
    threads.min(tasks).clamp(1, pool::MAX_THREADS)
}

/// How the packed driver splits an `m` by `n` by `k` product (after any swap), from the
/// shape and the thread count alone. The driver and [`threads_used`] both read it, so the
/// reported thread count is the one the driver uses.
#[derive(Clone, Copy)]
struct Plan {
    /// The thread count passed to the pool: 1 below [`SINGLE_THREAD_FLOPS`].
    threads: usize,
    /// The length of every block of the sum but the last, which may be shorter. Blocks are
    /// of equal size and do not depend on the thread count.
    kc: usize,
    /// The width of every block of columns but the last, a multiple of `NR`.
    nc: usize,
    /// The compute grid of a full-width block of columns.
    full: Grid,
    /// The compute grid of the last block of columns.
    last: Grid,
}

impl Plan {
    /// The plan for kernel `K`; `m`, `n` and `k` are all non-zero.
    fn new<K: Kernel>(m: usize, n: usize, k: usize, threads: usize) -> Plan {
        let threads = threads_for(m, n, k, threads);
        let kc = k.div_ceil(k.div_ceil(K::KC));
        let nc = n.div_ceil(n.div_ceil(K::NC)).next_multiple_of(K::NR);
        let grid = |width| Grid::new(m, width, K::MR, K::NR, K::MC, threads);
        Plan {
            threads,
            kc,
            nc,
            full: grid(nc.min(n)),
            last: grid(n - (n - 1) / nc * nc),
        }
    }

    /// The most threads any `pool::run` of the call uses: packing runs one task per `NR`
    /// columns of a block, computing runs the block's grid.
    fn threads_used<K: Kernel>(&self, n: usize) -> usize {
        let last_width = n - (n - 1) / self.nc * self.nc;
        [(self.full, self.nc.min(n)), (self.last, last_width)]
            .into_iter()
            .map(|(grid, width)| launch(self.threads, grid.tasks().max(width.div_ceil(K::NR))))
            .max()
            .unwrap_or(1)
    }
}

/// The greatest common divisor of `a` and `b` (Euclid's algorithm); `gcd(a, 0)` is `a`.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// One cache line of floats: the unit the packing buffers are allocated in, so every
/// buffer, and every per-task slot in one, starts on a 64-byte boundary.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([f32; 16]);

/// Packing buffers, kept per calling thread and reused by every call it makes.
struct Scratch {
    /// The packed `kc` by `nc` block of `B`, shared read-only by the compute tasks.
    b: Vec<Line>,
    /// The packed rows of `A`, one slot per thread of the call.
    a: Vec<Line>,
}

thread_local! {
    /// This thread's packing buffers. A call borrows them for its whole duration and hands
    /// raw pointers into them to its tasks; they grow on first use and are then reused.
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch {
            b: Vec::new(),
            a: Vec::new(),
        })
    };
}

/// Grows `v` to hold at least `lines` lines; never shrinks it.
fn reserve(v: &mut Vec<Line>, lines: usize) {
    if v.len() < lines {
        v.resize(lines, Line([0.0; 16]));
    }
}

/// A raw pointer that tasks on several threads may hold at once. Each use states which
/// part of the memory a task touches and why no other task touches it at the same time.
#[derive(Clone, Copy)]
struct Shared(*mut f32);

// SAFETY: `Shared` is only a pointer; every dereference is in an `unsafe` block of this
// module whose comment shows that tasks running at the same time read or write disjoint
// memory (or only read), and that the memory outlives the `pool::run` call that uses it.
unsafe impl Sync for Shared {}

impl Shared {
    /// The pointer. A method rather than a field access, so that closures capture the whole
    /// `Shared` (which is `Sync`) instead of the bare pointer.
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

/// How one `kc` by `nc` block of the product is split into compute tasks.
#[derive(Clone, Copy)]
pub(super) struct Grid {
    /// `mr`-row panels of `A` per task.
    m_panels: usize,
    /// Number of task rows.
    m_tasks: usize,
    /// `nr`-column panels of `B` per task.
    n_panels: usize,
    /// Number of task columns.
    n_tasks: usize,
}

impl Grid {
    /// Splits `m` rows and `n` columns into tasks of whole `mr` by `nr` panels, at most
    /// `mc` rows each, aiming for [`TASKS_PER_THREAD`] tasks per thread when `threads > 1`
    /// (splitting columns first, then rows) and one column of tasks on one thread. When the
    /// `mc` limit leaves more tasks than threads, the rows are cut into more tasks, to a
    /// whole number per thread where the panels allow it, so that no thread runs a last task
    /// alone while the others wait. Which task computes a tile never changes the tile's
    /// value, so the split may depend on the thread count.
    pub(super) fn new(m: usize, n: usize, mr: usize, nr: usize, mc: usize, threads: usize) -> Grid {
        let rows = m.div_ceil(mr);
        let cols = n.div_ceil(nr);
        let mut m_tasks = rows.div_ceil(mc / mr);
        let want = if threads == 1 {
            1
        } else {
            threads * TASKS_PER_THREAD
        };
        let n_tasks = want.div_ceil(m_tasks).min(cols);
        if m_tasks * n_tasks < want {
            m_tasks = want.div_ceil(n_tasks).min(rows).max(m_tasks);
        }
        if threads > 1 && m_tasks * n_tasks > threads {
            // The fewest task rows at or above `m_tasks` that, once the rows are shared out
            // in whole panels, make the count a multiple of `threads`; none if the rows do
            // not allow it.
            let step = threads / gcd(n_tasks, threads);
            let mut rows_tasks = m_tasks.next_multiple_of(step);
            while rows_tasks <= rows {
                if (rows.div_ceil(rows.div_ceil(rows_tasks)) * n_tasks).is_multiple_of(threads) {
                    m_tasks = rows_tasks;
                    break;
                }
                rows_tasks += step;
            }
        }
        let m_panels = rows.div_ceil(m_tasks);
        let n_panels = cols.div_ceil(n_tasks);
        Grid {
            m_panels,
            m_tasks: rows.div_ceil(m_panels),
            n_panels,
            n_tasks: cols.div_ceil(n_panels),
        }
    }

    /// The number of tasks.
    pub(super) fn tasks(&self) -> usize {
        self.m_tasks * self.n_tasks
    }
}

/// Packs rows `i0..i1` of `A`, columns `p0..p0 + kc`, into `dst` as `mr`-row panels: each
/// panel holds `kc` groups of `mr` values (one per row), each value written `REP` times in a
/// row (so a kernel can load it already broadcast), rows past `i1` padded with zeros.
/// `panel` packs one full panel faster when it can (it returns whether it did); the rest
/// go through [`pack_a_panel`].
pub(super) fn pack_a<const REP: usize>(
    a: &MatRef<f32>,
    (i0, i1): (usize, usize),
    (p0, kc): (usize, usize),
    mr: usize,
    dst: &mut [f32],
    panel: impl Fn(&MatRef<f32>, usize, (usize, usize), &mut [f32]) -> bool,
) {
    for (index, out) in dst.chunks_exact_mut(mr * kc * REP).enumerate() {
        let r0 = i0 + index * mr;
        if r0 >= i1 {
            break;
        }
        let rows = mr.min(i1 - r0);
        if rows < mr || !panel(a, r0, (p0, kc), out) {
            pack_a_panel::<REP>(a, (r0, rows), (p0, kc), mr, out);
        }
    }
}

/// Packs one panel of [`pack_a`]: rows `r0..r0 + rows` (`rows <= mr`), padded to `mr`.
fn pack_a_panel<const REP: usize>(
    a: &MatRef<f32>,
    (r0, rows): (usize, usize),
    (p0, kc): (usize, usize),
    mr: usize,
    out: &mut [f32],
) {
    let (rs, cs) = (a.row_stride, a.col_stride);
    // Where each row of the panel starts in `A`, at column `p0`.
    let mut starts = [0usize; MAX_MR];
    for (r, s) in starts.iter_mut().enumerate().take(rows) {
        *s = (r0 + r) * rs + p0 * cs;
    }
    // Written in order, one group of `mr · REP` values per step of k.
    for (p, group) in out.chunks_exact_mut(mr * REP).enumerate().take(kc) {
        for (r, vals) in group.as_chunks_mut::<REP>().0.iter_mut().enumerate() {
            let v = if r < rows {
                a.data[starts[r] + p * cs]
            } else {
                0.0
            };
            *vals = [v; REP];
        }
    }
}

/// The most rows a kernel's tile may have.
const MAX_MR: usize = 16;

/// Packs columns `j0..j0 + cols` of `B` (`cols <= nr`), rows `p0..p0 + kc`, into `dst` as
/// one panel: `kc` groups of `nr` values, columns past `cols` padded with zeros.
pub(super) fn pack_b(
    b: &MatRef<f32>,
    (j0, cols): (usize, usize),
    (p0, kc): (usize, usize),
    nr: usize,
    dst: &mut [f32],
) {
    let (rs, cs) = (b.row_stride, b.col_stride);
    if cs == 1 {
        // Each row is contiguous along n.
        for p in 0..kc {
            let start = (p0 + p) * rs + j0;
            let group = &mut dst[p * nr..(p + 1) * nr];
            group[..cols].copy_from_slice(&b.data[start..start + cols]);
            group[cols..].fill(0.0);
        }
    } else if rs == 1 {
        // Each column is contiguous along k (B is a transposed row-major matrix).
        for j in 0..nr {
            if j < cols {
                let start = (j0 + j) * cs + p0;
                let col = &b.data[start..start + kc];
                for (p, &v) in col.iter().enumerate() {
                    dst[p * nr + j] = v;
                }
            } else {
                for p in 0..kc {
                    dst[p * nr + j] = 0.0;
                }
            }
        }
    } else {
        for p in 0..kc {
            for j in 0..nr {
                dst[p * nr + j] = if j < cols {
                    b.data[(p0 + p) * rs + (j0 + j) * cs]
                } else {
                    0.0
                };
            }
        }
    }
}

/// The packed product with kernel `K` on `threads` threads, for `m`, `n` and `k` all
/// non-zero and shapes already checked.
///
/// # Safety
///
/// The CPU must support `K`'s instruction set.
unsafe fn packed<K: Kernel>(
    threads: usize,
    alpha: f32,
    a: MatRef<f32>,
    b: MatRef<f32>,
    beta: f32,
    c: MatMut<f32>,
) {
    const {
        assert!(
            K::NR % 4 == 0
                && K::A_REP >= 1
                && K::MR <= MAX_MR
                && K::MC % K::MR == 0
                && K::MC >= K::MR
        )
    };
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    let (mr, nr) = (K::MR, K::NR);
    let plan = Plan::new::<K>(m, n, k, threads);
    let Plan {
        threads,
        kc,
        nc,
        full,
        last,
    } = plan;
    let (crs, ccs) = (c.row_stride, c.col_stride);
    // The one raw pointer to C, taken from the `MatMut` borrowed for the whole call.
    let c_ptr = Shared(c.data.as_mut_ptr());
    SCRATCH.with_borrow_mut(|scratch| {
        // Sizes depend only on the shape and the thread count, so a warmed-up caller never
        // grows them again. `full` and `last` bound the tasks and rows of every block, and
        // no compute task runs on a worker numbered `slots` or above.
        let slots = launch(threads, full.tasks().max(last.tasks()));
        let a_slot = (full.m_panels.max(last.m_panels) * mr * kc * K::A_REP).div_ceil(16);
        reserve(&mut scratch.b, (kc * nc).div_ceil(16));
        reserve(&mut scratch.a, slots * a_slot);
        let b_buf = Shared(scratch.b.as_mut_ptr().cast::<f32>());
        let a_buf = Shared(scratch.a.as_mut_ptr().cast::<f32>());
        for j0 in (0..n).step_by(nc) {
            let ncur = nc.min(n - j0);
            let panels = ncur.div_ceil(nr);
            let grid = Grid::new(m, ncur, mr, nr, K::MC, threads);
            for p0 in (0..k).step_by(kc) {
                let kcur = kc.min(k - p0);
                let beta = if p0 == 0 { beta } else { 1.0 };
                pool::run(launch(threads, panels), panels, &|t: pool::Task| {
                    let j = t.index * nr;
                    // SAFETY: panel `t.index` occupies floats `t.index·kcur·nr` up to
                    // `(t.index + 1)·kcur·nr` of the B buffer, which holds `kc·nc` floats
                    // (`panels·nr <= nc`, `kcur <= kc`); each index runs once, so tasks
                    // write disjoint panels, and the buffer outlives this `run`.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            b_buf.ptr().add(t.index * kcur * nr),
                            kcur * nr,
                        )
                    };
                    K::pack_b(&b, (j0 + j, nr.min(ncur - j)), (p0, kcur), dst);
                });
                pool::run(
                    launch(threads, grid.tasks()),
                    grid.tasks(),
                    &|t: pool::Task| {
                        assert!(t.worker < slots, "gemm: worker {} of {slots}", t.worker);
                        let (ti, tj) = (t.index / grid.n_tasks, t.index % grid.n_tasks);
                        let i0 = ti * grid.m_panels * mr;
                        let i1 = m.min(i0 + grid.m_panels * mr);
                        let a_panels = (i1 - i0).div_ceil(mr);
                        // SAFETY: slot `t.worker` is floats `t.worker·a_slot·16` onward, `a_slot`
                        // lines long, inside the A buffer (`t.worker < slots`); it holds
                        // `a_panels·mr·kcur <= grid.m_panels·mr·kc` floats, at most `a_slot`
                        // lines. Two tasks
                        // running at the same time have different worker numbers, and the
                        // buffer outlives this `run`.
                        let a_pack = unsafe {
                            std::slice::from_raw_parts_mut(
                                a_buf.ptr().add(t.worker * a_slot * 16),
                                a_panels * mr * kcur * K::A_REP,
                            )
                        };
                        K::pack_a(&a, (i0, i1), (p0, kcur), a_pack);
                        let first = tj * grid.n_panels;
                        let last = panels.min(first + grid.n_panels);
                        // One A panel stays in L1 while the task's B panels stream past it.
                        for ip in 0..a_panels {
                            let i = i0 + ip * mr;
                            let rows = mr.min(m - i);
                            for jp in first..last {
                                let j = jp * nr;
                                let cols = nr.min(ncur - j);
                                // SAFETY: inside the B buffer, packed by the previous `run`, which
                                // finished before this one started; only read from now on.
                                let b_panel =
                                    unsafe { b_buf.ptr().add(jp * kcur * nr).cast_const() };
                                let dest = Dest {
                                    // SAFETY: (i, j0 + j) is an element of C (i < m,
                                    // j0 + j < n), so its offset is inside C's slice.
                                    ptr: unsafe { c_ptr.ptr().add(i * crs + (j0 + j) * ccs) },
                                    rs: crs,
                                    cs: ccs,
                                    rows,
                                    cols,
                                };
                                // SAFETY: the A panel holds `mr·A_REP·kcur` floats, the B panel
                                // `nr·kcur`. The slots and the B buffer start on 64-byte
                                // boundaries and panels at multiples of `nr·kcur` and
                                // `mr·A_REP·kcur` floats, so B panels are 16-byte aligned
                                // (`nr` is a multiple of 4, checked above) and A panels are
                                // when `mr·A_REP` is a multiple of 4, as `Kernel::tile` states.
                                // The tile's valid elements lie in C (rows and cols are clipped
                                // to m and n), and this task is the only one writing rows
                                // i0..i1 of columns of panels first..last; `MatMut` guarantees
                                // distinct elements do not alias. The CPU supports `K` (this
                                // function's contract).
                                unsafe {
                                    K::tile(
                                        kcur,
                                        a_pack.as_ptr().add(ip * mr * kcur * K::A_REP),
                                        b_panel,
                                        alpha,
                                        beta,
                                        &dest,
                                    );
                                }
                            }
                        }
                    },
                );
            }
        }
    });
}
