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

use std::cell::RefCell;

use super::{Level, MatMut, MatRef, detected_features, pool, reference, supported};

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
    /// groups of `MR` values, each repeated `A_REP` times) and `B̃` the packed `B` panel (`kc` groups of `NR`
    /// values). Each element's sum runs over `p` in order from zero, with a separate multiply
    /// and add, and is then stored as `alpha · sum` when `beta` is 0 (C is not read) or
    /// `alpha · sum + beta · c` otherwise.
    ///
    /// # Safety
    ///
    /// `a` must be valid for reading `MR · A_REP · kc` values and `b` for reading `NR · kc`
    /// values, both aligned to 16 bytes. `dest.rows <= MR`, `dest.cols <= NR`, and for every
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
/// calling thread alone, with the same blocking. An estimate, not tuned: about 6 µs of work
/// on one core, against two thread-pool round trips of about 0.5 µs each.
const SINGLE_THREAD_FLOPS: usize = 1 << 17;

/// Compute tasks per thread the driver aims for. More tasks balance uneven work better but
/// pack each row of `A` more often; one was fastest when measured (see the SSE kernel's
/// notes in `x86.rs`).
const TASKS_PER_THREAD: usize = 1;

/// General matrix multiply in f32, `C = alpha · A · B + beta · C`, with the kernel for the
/// current [`level`](super::level) on [`threads`](super::threads) threads.
///
/// Results match [`reference::gemm`] within the rounding of a different summation order
/// (the sum over `k` runs in blocks), are bit-identical for any thread count, and are the
/// same from run to run at the same level. When `beta` is zero the old contents of `C` are
/// never read, so NaN in `C` does not leak into the result.
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
/// [`Level::Sse42`] runs the packed SSE4.2 kernel. [`Level::Avx2Fma`] runs the same SSE4.2
/// kernel for now (every CPU with AVX2 has SSE4.2), and [`Level::Neon`] the reference loops.
/// [`Level::Reference`] runs [`reference::gemm`] on one thread. Products too small to be
/// worth splitting run on one thread; [`threads_used`] tells how many a call uses.
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
        "gemm_f32_with: this CPU cannot run kernel level {level} (detected features: {})",
        detected_features()
    );
    reference::check_gemm_shapes(&a, &b, &c);
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    if m == 0 || n == 0 || k == 0 {
        // Nothing to multiply: C is untouched (m or n is 0) or becomes beta · C (k is 0).
        return reference::gemm(alpha, a, b, beta, c);
    }
    match level {
        // AVX2 CPUs run the SSE4.2 kernel until there is an AVX2 one; the check keeps that
        // sound even on a CPU that reported AVX2 without SSE4.2.
        #[cfg(target_arch = "x86_64")]
        Level::Sse42 | Level::Avx2Fma if supported(Level::Sse42) => {
            let threads = threads_for(m, n, k, threads);
            if swaps(m, n) {
                // Cᵀ = Bᵀ · Aᵀ: the same sums of the same products in the same order (a
                // product of two floats does not depend on their order), so the same bits.
                // SAFETY: this CPU supports SSE4.2 (checked just above).
                unsafe {
                    packed::<super::x86::Sse>(
                        threads,
                        alpha,
                        b.transpose(),
                        a.transpose(),
                        beta,
                        c.transpose(),
                    );
                }
            } else {
                // SAFETY: this CPU supports SSE4.2 (checked just above).
                unsafe { packed::<super::x86::Sse>(threads, alpha, a, b, beta, c) }
            }
        }
        _ => reference::gemm(alpha, a, b, beta, c),
    }
}

/// The number of threads [`gemm_f32_with`] uses for an `m` by `k` times `k` by `n` product
/// at `level` when given `threads`: 1 for the reference loops and for products too small to
/// be worth splitting, else `threads`.
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
    match level {
        Level::Sse42 | Level::Avx2Fma
            if cfg!(target_arch = "x86_64") && supported(Level::Sse42) =>
        {
            threads_for(m, n, k, threads)
        }
        _ => 1,
    }
}

/// Whether the packed driver computes `Cᵀ = Bᵀ · Aᵀ` instead: when `C` is much taller than
/// wide. The kernel stores each value of `A` four times over, so packing costs most when `A`
/// is the large operand and every packed row serves only a few columns.
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
struct Grid {
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
    /// (splitting columns first, then rows) and one column of tasks on one thread. Which
    /// task computes a tile never changes the tile's value, so the split may depend on the
    /// thread count.
    fn new(m: usize, n: usize, mr: usize, nr: usize, mc: usize, threads: usize) -> Grid {
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
    fn tasks(&self) -> usize {
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
                && (K::MR * K::A_REP) % 4 == 0
                && K::MR <= MAX_MR
                && K::MC % K::MR == 0
                && K::MC >= K::MR
        )
    };
    let (m, n, k) = (a.rows(), b.cols(), a.cols());
    let (mr, nr) = (K::MR, K::NR);
    // Blocks of equal size (the last may be shorter by less than one step): the same for any
    // thread count.
    let kc = k.div_ceil(k.div_ceil(K::KC));
    let nc = n.div_ceil(n.div_ceil(K::NC)).next_multiple_of(nr);
    let (crs, ccs) = (c.row_stride, c.col_stride);
    // The one raw pointer to C, taken from the `MatMut` borrowed for the whole call.
    let c_ptr = Shared(c.data.as_mut_ptr());
    SCRATCH.with_borrow_mut(|scratch| {
        // Sizes depend only on the shape and the thread count, so a warmed-up caller never
        // grows them again.
        // Every block of columns is `nc` wide except perhaps the last, so these two grids
        // bound the tasks and rows of every block.
        let full = Grid::new(m, nc.min(n), mr, nr, K::MC, threads);
        let last = Grid::new(m, n - (n - 1) / nc * nc, mr, nr, K::MC, threads);
        let slots = threads.min(full.tasks().max(last.tasks()));
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
                pool::run(threads, panels, &|t: pool::Task| {
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
                pool::run(threads, grid.tasks(), &|t: pool::Task| {
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
                            let b_panel = unsafe { b_buf.ptr().add(jp * kcur * nr).cast_const() };
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
                            // `nr·kcur`, both 16-byte aligned (the slots and the B buffer
                            // start on 64-byte boundaries, and panels at multiples of
                            // `nr·kcur` and `mr·A_REP·kcur` floats, multiples of 4 since
                            // `nr` and `A_REP` are checked or set to be).
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
                });
            }
        }
    });
}
