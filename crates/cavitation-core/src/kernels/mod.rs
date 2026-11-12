//! CPU kernels: strided matrix views, the plain reference implementations, the kernel level
//! (which instruction set, or which vendor library, the fast kernels use) and the thread count.
//!
//! The tape's matrix products and the optimisers run on these kernels; most programs never
//! call them directly. [`MatRef`] and [`MatMut`] describe a matrix inside a slice by its
//! strides, so a transposed or column-major matrix needs no copy. [`gemm`] and [`matvec`]
//! hold the fast kernels, [`reference`](mod@reference) the plain loops they are tested against, and
//! [`pool`] the thread pool the threaded kernels share. [`level`] and [`threads`] report
//! what the fast kernels use, and `CAVITATION_KERNEL` and `CAVITATION_THREADS` override them.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::{MatRef, level, supported, threads};
//! // The current level is always one this CPU can run, and there is at least one thread.
//! assert!(supported(level()));
//! assert!(threads() >= 1);
//! // A 2 x 3 row-major matrix, and its transpose read through the same storage.
//! let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let m = MatRef::new(&a, 2, 3, 3, 1).unwrap();
//! let t = MatRef::new(&a, 3, 2, 1, 3).unwrap();
//! assert_eq!((m.get(0, 2), t.get(2, 0)), (3.0, 3.0));
//! ```

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(any(test, all(target_vendor = "apple", target_arch = "aarch64")))]
mod accelerate;
pub mod gemm;
pub mod matvec;
pub mod pool;
pub mod reference;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(test)]
#[path = "tests.rs"]
mod twin_tests;

use crate::Real;
use crate::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// The instruction set, or the vendor library, the fast kernels use.
///
/// The level starts as [`detected`] (or as `CAVITATION_KERNEL`, if set) and can be changed
/// with [`set_level`]. Every level gives results within the documented tolerance of the
/// reference kernels; the same level and inputs give bit-identical results.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::Level;
/// assert_eq!(Level::Avx2Fma.to_string(), "avx2");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// The plain reference loops, on every CPU.
    Reference = 0,
    /// x86_64 with SSE4.2: 4-wide f32 vectors.
    Sse42 = 1,
    /// x86_64 with AVX2 and FMA: 8-wide f32 vectors with fused multiply-add.
    Avx2Fma = 2,
    /// aarch64 with NEON: 4-wide f32 vectors with fused multiply-add.
    Neon = 3,
    /// Apple Silicon: matrix products through Apple's Accelerate framework (`cblas_sgemm`),
    /// everything else through the NEON kernels. Products whose layout BLAS cannot express
    /// (neither stride 1, overlapping or broadcast rows, sizes beyond `i32`) and products
    /// with `alpha = 0` run the NEON GEMM. Results match the reference within the GEMM
    /// tolerance, as every level does, and differ from NEON's in the last bits.
    Accelerate = 4,
}

impl Level {
    /// Every level, in the order of their numbers.
    const ALL: [Level; 5] = [
        Level::Reference,
        Level::Sse42,
        Level::Avx2Fma,
        Level::Neon,
        Level::Accelerate,
    ];

    /// The level stored as `n` in [`LEVEL`], if `n` is one.
    fn from_u8(n: u8) -> Option<Level> {
        Level::ALL.get(usize::from(n)).copied()
    }
}

impl fmt::Display for Level {
    /// Writes the name `CAVITATION_KERNEL` accepts: `reference`, `sse4.2`, `avx2`, `neon` or
    /// `accelerate`.
    ///
    /// ```
    /// use cavitation_core::kernels::Level;
    /// assert_eq!(Level::Sse42.to_string(), "sse4.2");
    /// assert_eq!(Level::Accelerate.to_string(), "accelerate");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Level::Reference => "reference",
            Level::Sse42 => "sse4.2",
            Level::Avx2Fma => "avx2",
            Level::Neon => "neon",
            Level::Accelerate => "accelerate",
        })
    }
}

/// Parses a level name as `CAVITATION_KERNEL` takes it: `reference`, `sse4.2`, `avx2`, `neon`
/// or `accelerate`. Whether this CPU can run the level is not checked; see [`supported`].
///
/// # Errors
///
/// Returns a message naming `s` and the accepted names if `s` is not one of them.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{parse_level, Level};
/// assert_eq!(parse_level("avx2"), Ok(Level::Avx2Fma));
/// assert_eq!(parse_level("accelerate"), Ok(Level::Accelerate));
/// assert!(parse_level("avx512").is_err());
/// ```
pub fn parse_level(s: &str) -> Result<Level, String> {
    Level::ALL
        .into_iter()
        .find(|l| l.to_string() == s)
        .ok_or_else(|| {
            format!(
                "unknown kernel level `{s}`; expected reference, sse4.2, avx2, neon or accelerate"
            )
        })
}

/// Parses a thread count as `CAVITATION_THREADS` takes it: a positive integer.
///
/// # Errors
///
/// Returns a message naming `s` if it is not a positive integer (for example `0`, `-1` or
/// `abc`).
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::parse_threads;
/// assert_eq!(parse_threads("4"), Ok(4));
/// assert!(parse_threads("0").is_err());
/// ```
pub fn parse_threads(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err(format!(
            "expected a positive integer thread count, got `{s}`"
        )),
    }
}

/// Whether this CPU can run `level`. [`Level::Reference`] always can;
/// [`Level::Accelerate`] runs on Apple Silicon only (an Apple aarch64 target, where the
/// Accelerate framework is part of the system).
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{supported, Level};
/// assert!(supported(Level::Reference));
/// ```
pub fn supported(level: Level) -> bool {
    match level {
        Level::Reference => true,
        #[cfg(target_arch = "x86_64")]
        Level::Sse42 => std::arch::is_x86_feature_detected!("sse4.2"),
        #[cfg(target_arch = "x86_64")]
        Level::Avx2Fma => {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(target_arch = "aarch64")]
        Level::Neon => std::arch::is_aarch64_feature_detected!("neon"),
        // The level runs the NEON kernels for everything but the matrix products.
        #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
        Level::Accelerate => std::arch::is_aarch64_feature_detected!("neon"),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// The best level this CPU supports: AVX2 with FMA, then SSE4.2, on x86_64; Accelerate on
/// Apple Silicon; NEON on other aarch64 systems; the reference kernels anywhere else.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{detected, supported};
/// assert!(supported(detected()));
/// ```
pub fn detected() -> Level {
    [Level::Accelerate, Level::Avx2Fma, Level::Neon, Level::Sse42]
        .into_iter()
        .find(|&l| supported(l))
        .unwrap_or(Level::Reference)
}

/// The SIMD features this CPU reports, as a comma-separated list for error messages.
fn detected_features() -> String {
    let mut found: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.2") {
            found.push("sse4.2");
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            found.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            found.push("fma");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            found.push("neon");
        }
    }
    #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
    found.push("accelerate");
    if found.is_empty() {
        "none".to_string()
    } else {
        found.join(", ")
    }
}

/// The current level as its number, or [`UNSET`] before the first use.
static LEVEL: AtomicU8 = AtomicU8::new(UNSET);

/// The value of [`LEVEL`] before the level is first read or set.
const UNSET: u8 = u8::MAX;

/// The level `CAVITATION_KERNEL` names, or [`detected`] if it is not set.
///
/// # Panics
///
/// Panics if the variable is not a level name, or names a level this CPU cannot run.
fn level_from_env() -> Level {
    let Some(value) = std::env::var_os("CAVITATION_KERNEL") else {
        return detected();
    };
    let value = value.to_string_lossy();
    let level = parse_level(&value).unwrap_or_else(|e| panic!("CAVITATION_KERNEL: {e}"));
    assert!(
        supported(level),
        "CAVITATION_KERNEL=`{level}`: this CPU cannot run kernel level {level} \
         (detected features: {})",
        detected_features()
    );
    level
}

/// The current kernel level: the last one passed to [`set_level`], else `CAVITATION_KERNEL`
/// if it is set, else [`detected`]. Always a level this CPU supports.
///
/// # Panics
///
/// On first use, if `CAVITATION_KERNEL` is set to something other than a level name, or to a
/// level this CPU cannot run; the message names the level and the detected features.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{level, supported};
/// assert!(supported(level()));
/// ```
pub fn level() -> Level {
    if let Some(level) = Level::from_u8(LEVEL.load(Ordering::Relaxed)) {
        return level;
    }
    let level = level_from_env();
    // A concurrent first use or `set_level` may have stored a level first; keep theirs.
    match LEVEL.compare_exchange(UNSET, level as u8, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => level,
        Err(current) => Level::from_u8(current).unwrap_or(level),
    }
}

/// Makes `level` the current kernel level for the whole process.
///
/// # Errors
///
/// Returns [`Error::Unsupported`], naming the level and the detected features, if this CPU
/// cannot run `level`; the current level is then unchanged.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{level, set_level, Level};
/// set_level(Level::Reference).unwrap();
/// assert_eq!(level(), Level::Reference);
/// ```
pub fn set_level(level: Level) -> Result<(), Error> {
    if !supported(level) {
        return Err(Error::Unsupported(format!(
            "kernel level {level} on this CPU (detected features: {})",
            detected_features()
        )));
    }
    LEVEL.store(level as u8, Ordering::Relaxed);
    Ok(())
}

/// The current thread count, or 0 before the first use.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// The thread count `CAVITATION_THREADS` gives, or [`default_threads`] if it is not set.
///
/// # Panics
///
/// Panics, naming the variable and its value, if it is not a positive integer.
fn threads_from_env() -> usize {
    match std::env::var_os("CAVITATION_THREADS") {
        Some(value) => parse_threads(&value.to_string_lossy())
            .unwrap_or_else(|e| panic!("CAVITATION_THREADS: {e}")),
        None => default_threads(),
    }
}

/// [`std::thread::available_parallelism`], or 1 if it is unknown.
fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// The thread count used when neither [`set_threads`] nor `CAVITATION_THREADS` gives one:
/// the number of performance cores where the system reports it (Apple Silicon), else
/// [`std::thread::available_parallelism`] (1 if that is unknown). [`threads`] calls it once,
/// on first use, and keeps the answer.
///
/// On an Apple M4 (4 performance and 6 efficiency cores) training at 10 threads was slower
/// than at 4: 1.14 against 0.97 s per MNIST epoch at batch 128, and 392k against 677k digits
/// samples per second (measured with `cavitation-bench all`, NEON kernels, medians of 3 runs).
///
/// # Examples
///
/// The function is private; its answer is what [`threads`] returns when nothing overrides it:
///
/// ```
/// let n = cavitation_core::kernels::threads();
/// assert!(n >= 1);
/// if std::env::var_os("CAVITATION_THREADS").is_none() {
///     // Never more threads than the machine runs at once.
///     assert!(n <= std::thread::available_parallelism().map_or(1, |n| n.get()));
/// }
/// ```
fn default_threads() -> usize {
    default_threads_from(performance_cores)
}

/// [`default_threads`] with the performance-core count read by `read`, so tests can stand in
/// for the system: `read`'s count if it gives a positive one (at most the available
/// parallelism), else the available parallelism (1 if that is unknown).
fn default_threads_from(read: impl FnOnce() -> std::io::Result<usize>) -> usize {
    let available = available_threads();
    match read() {
        Ok(n) if n > 0 => n.min(available),
        _ => available,
    }
}

/// The number of performance cores, from `sysctl hw.perflevel0.physicalcpu`. Fails on a Mac
/// without performance levels (Intel), and the error says why.
#[cfg(target_vendor = "apple")]
fn performance_cores() -> std::io::Result<usize> {
    use std::ffi::{c_char, c_int, c_void};
    // SAFETY: the declaration matches `sysctlbyname` in the system's `<sys/sysctl.h>`
    // (`size_t` is `usize` on every Apple target), and the C library exports it.
    unsafe extern "C" {
        /// Reads (or, with `newp` non-null, writes) the system value named `name` into the
        /// `*oldlenp` bytes at `oldp`, storing the length written in `*oldlenp`. Returns 0
        /// on success, or -1 with `errno` set.
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut value: i32 = 0;
    let mut len = size_of::<i32>();
    // SAFETY: the name is a NUL-terminated C string; `oldp` points at `len` writable bytes
    // (an `i32` this function owns), and `oldlenp` at `len`; nothing is written (`newp` null,
    // `newlen` 0). All three pointers outlive the call.
    let status = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if len != size_of::<i32>() {
        return Err(std::io::Error::other(format!(
            "hw.perflevel0.physicalcpu: expected 4 bytes, got {len}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        std::io::Error::other(format!("hw.perflevel0.physicalcpu: negative count {value}"))
    })
}

/// The number of performance cores: unknown on systems other than Apple's, so the default
/// thread count falls back to the available parallelism.
#[cfg(not(target_vendor = "apple"))]
fn performance_cores() -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "performance cores are only read on Apple systems",
    ))
}

/// The number of threads the threaded kernels use: the last value passed to
/// [`set_threads`], else `CAVITATION_THREADS` if it is set, else the number of performance
/// cores on Apple Silicon (where the efficiency cores slow training down) and
/// [`std::thread::available_parallelism`] elsewhere (1 if that is unknown). Always at
/// least 1.
///
/// # Panics
///
/// On first use, if `CAVITATION_THREADS` is set to anything but a positive integer; the
/// message names the variable and its value.
///
/// # Examples
///
/// ```
/// assert!(cavitation_core::kernels::threads() >= 1);
/// ```
pub fn threads() -> usize {
    let n = THREADS.load(Ordering::Relaxed);
    if n != 0 {
        return n;
    }
    let n = threads_from_env();
    // A concurrent first use or `set_threads` may have stored a count first; keep theirs.
    match THREADS.compare_exchange(0, n, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => n,
        Err(current) => current,
    }
}

/// Sets the number of threads the threaded kernels use, for the whole process. Results do
/// not depend on it: each output element is computed by one thread in a fixed order.
///
/// Any count is accepted, but one call into the thread pool uses at most 256 threads; a
/// larger count runs on 256, with the same results.
///
/// # Panics
///
/// Panics if `n` is 0.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::{set_threads, threads};
/// set_threads(2);
/// assert_eq!(threads(), 2);
/// ```
pub fn set_threads(n: usize) {
    assert!(
        n >= 1,
        "set_threads: the thread count must be at least 1, got 0"
    );
    THREADS.store(n, Ordering::Relaxed);
}

/// Returns the number of elements a `rows` by `cols` view with the given strides needs
/// `data` to hold, or `None` if the arithmetic overflows. Empty views need none.
fn required_len(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    (rows - 1)
        .checked_mul(row_stride)?
        .checked_add((cols - 1).checked_mul(col_stride)?)?
        .checked_add(1)
}

/// Whether no two distinct elements of a `rows` by `cols` view can share an offset.
///
/// An empty view (no rows or no columns) always passes. Otherwise only dimensions with
/// extent above 1 count. A single such dimension needs a stride of at least 1. Two need, with the smaller stride `s` and its extent `e`, `s >= 1` and
/// `e * s <= ` the larger stride. This is sufficient, not necessary.
fn is_non_aliasing(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> bool {
    if rows == 0 || cols == 0 {
        // No elements, so none can share an offset.
        return true;
    }
    // (extent, stride) of each dimension longer than 1, without allocating.
    let row = (rows > 1).then_some((rows, row_stride));
    let col = (cols > 1).then_some((cols, col_stride));
    match (row, col) {
        (None, None) => true,
        (Some((_, stride)), None) | (None, Some((_, stride))) => stride >= 1,
        (Some(a), Some(b)) => {
            let (small, large) = if a.1 <= b.1 { (a, b) } else { (b, a) };
            small.1 >= 1 && small.0.checked_mul(small.1).is_some_and(|v| v <= large.1)
        }
    }
}

/// A read-only matrix view over a slice, with an independent stride for rows and columns.
///
/// Element `(i, j)` is `data[i * row_stride + j * col_stride]`. A row-major `r` by `c`
/// matrix has strides `(c, 1)`; its transpose is the same data with the strides swapped.
/// The constructor checks that every element lies inside the slice, so a view built from
/// safe code can never index out of bounds.
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::MatRef;
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let m = MatRef::new(&data, 2, 3, 3, 1).unwrap();
/// assert_eq!(m.get(1, 2), 6.0);
/// assert_eq!(m.transpose().get(2, 1), 6.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T> {
    /// The underlying storage.
    data: &'a [T],
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Distance in `data` between vertically adjacent elements.
    row_stride: usize,
    /// Distance in `data` between horizontally adjacent elements.
    col_stride: usize,
}

impl<'a, T: Real> MatRef<'a, T> {
    /// Builds a view, or returns `None` if the largest index
    /// `(rows - 1) * row_stride + (cols - 1) * col_stride` is not inside `data`.
    /// A view with zero rows or zero columns is always allowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let data = [0.0f32; 5];
    /// assert!(MatRef::new(&data, 2, 3, 3, 1).is_none()); // needs 6 values
    /// assert!(MatRef::new(&data, 0, 3, 3, 1).is_some());
    /// ```
    pub fn new(
        data: &'a [T],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let need = required_len(rows, cols, row_stride, col_stride)?;
        if need > data.len() {
            return None;
        }
        Some(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// The number of rows.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The distance in the slice between vertically adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.row_stride(), 3);
    /// ```
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// The distance in the slice between horizontally adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[0.0f32; 6], 2, 3, 3, 1).unwrap();
    /// assert_eq!(m.col_stride(), 1);
    /// ```
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// The element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[1.0f32, 2.0, 3.0, 4.0], 2, 2, 2, 1).unwrap();
    /// assert_eq!(m.get(1, 0), 3.0);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(
            i < self.rows && j < self.cols,
            "MatRef::get: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        self.data[i * self.row_stride + j * self.col_stride]
    }

    /// The same data seen as the transpose: rows and columns swap, and so do the strides.
    ///
    /// ```
    /// use cavitation_core::kernels::MatRef;
    /// let m = MatRef::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, 3, 1).unwrap();
    /// let t = m.transpose();
    /// assert_eq!((t.rows(), t.cols()), (3, 2));
    /// assert_eq!(t.get(0, 1), 4.0);
    /// ```
    pub fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }
}

/// A writable matrix view over a slice, with the same layout rules as [`MatRef`].
///
/// A `MatMut` never aliases: `new` returns `None` unless distinct elements map to distinct
/// positions in the slice, so writing different elements can never touch the same value.
/// An empty view (no rows or no columns) is always accepted. Otherwise only dimensions
/// longer than 1 count (a single row or column may have any stride). For
/// two such dimensions the rule is conservative: with the smaller stride `s` and its extent
/// `e`, it needs `s >= 1` and `e * s <=` the larger stride. A few non-aliasing layouts, such
/// as strides `(3, 2)` on a 2 x 2 view, are rejected too. A row-major, column-major or
/// transposed layout is always accepted. ([`MatRef`] allows overlap, since reading is safe.)
///
/// # Examples
///
/// ```
/// use cavitation_core::kernels::MatMut;
/// let mut data = [0.0f32; 4];
/// let mut m = MatMut::new(&mut data, 2, 2, 2, 1).unwrap();
/// *m.get_mut(1, 1) = 7.0;
/// assert_eq!(data, [0.0, 0.0, 0.0, 7.0]);
/// ```
#[derive(Debug)]
pub struct MatMut<'a, T> {
    /// The underlying storage.
    data: &'a mut [T],
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Distance in `data` between vertically adjacent elements.
    row_stride: usize,
    /// Distance in `data` between horizontally adjacent elements.
    col_stride: usize,
}

impl<'a, T: Real> MatMut<'a, T> {
    /// Builds a view, or returns `None` if the bounds rule of [`MatRef::new`] fails or two
    /// distinct elements could share a position (see the type's documentation).
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut data = [0.0f32; 5];
    /// assert!(MatMut::new(&mut data, 2, 3, 3, 1).is_none()); // out of bounds
    /// assert!(MatMut::new(&mut data, 2, 2, 0, 0).is_none()); // aliasing
    /// ```
    pub fn new(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        if !is_non_aliasing(rows, cols, row_stride, col_stride) {
            return None;
        }
        let need = required_len(rows, cols, row_stride, col_stride)?;
        if need > data.len() {
            return None;
        }
        Some(Self {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// The number of rows.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The distance in the slice between vertically adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().row_stride(), 3);
    /// ```
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// The distance in the slice between horizontally adjacent elements.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 6];
    /// assert_eq!(MatMut::new(&mut d, 2, 3, 3, 1).unwrap().col_stride(), 1);
    /// ```
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// The element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [1.0f32, 2.0];
    /// let m = MatMut::new(&mut d, 1, 2, 2, 1).unwrap();
    /// assert_eq!(m.get(0, 1), 2.0);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(
            i < self.rows && j < self.cols,
            "MatMut::get: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        self.data[i * self.row_stride + j * self.col_stride]
    }

    /// The same data seen as the transpose: rows and columns swap, and so do the strides.
    pub(crate) fn transpose(self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// A mutable reference to the element at row `i`, column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    ///
    /// ```
    /// use cavitation_core::kernels::MatMut;
    /// let mut d = [0.0f32; 2];
    /// let mut m = MatMut::new(&mut d, 1, 2, 2, 1).unwrap();
    /// *m.get_mut(0, 1) += 3.0;
    /// assert_eq!(m.get(0, 1), 3.0);
    /// ```
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        assert!(
            i < self.rows && j < self.cols,
            "MatMut::get_mut: index ({i}, {j}) outside a {} x {} view",
            self.rows,
            self.cols
        );
        &mut self.data[i * self.row_stride + j * self.col_stride]
    }
}

/// Tests for the kernel levels and the thread-count setting.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    /// Every level name parses to its level and prints back the same; unknown names are
    /// rejected with an error naming them.
    #[test]
    fn parse_level_names() {
        let names = [
            ("reference", Level::Reference),
            ("sse4.2", Level::Sse42),
            ("avx2", Level::Avx2Fma),
            ("neon", Level::Neon),
            ("accelerate", Level::Accelerate),
        ];
        for (name, level) in names {
            assert_eq!(parse_level(name), Ok(level));
            assert_eq!(level.to_string(), name);
        }
        let err = parse_level("avx512").unwrap_err();
        assert!(err.contains("avx512"), "{err}");
    }

    /// Forcing a level this CPU cannot run is an `Unsupported` error and changes nothing.
    #[test]
    fn unsupported_level_is_an_error() {
        let foreign = if cfg!(target_arch = "aarch64") {
            Level::Avx2Fma
        } else {
            Level::Neon
        };
        assert!(!supported(foreign));
        let before = level();
        assert!(matches!(set_level(foreign), Err(Error::Unsupported(_))));
        assert_eq!(level(), before);
        assert!(supported(Level::Reference));
        assert!(supported(detected()));
    }

    /// Accelerate is supported, and is the detected level, exactly on Apple Silicon; anywhere
    /// else forcing it is an `Unsupported` error naming it, like any level the CPU cannot run.
    #[test]
    fn accelerate_only_on_apple_silicon() {
        let apple_silicon = cfg!(all(target_vendor = "apple", target_arch = "aarch64"));
        assert_eq!(supported(Level::Accelerate), apple_silicon);
        assert_eq!(detected() == Level::Accelerate, apple_silicon);
        if !apple_silicon {
            let before = level();
            match set_level(Level::Accelerate) {
                Err(Error::Unsupported(msg)) => assert!(msg.contains("accelerate"), "{msg}"),
                other => panic!("expected Unsupported, got {other:?}"),
            }
            assert_eq!(level(), before);
        }
    }

    /// Zero, negative numbers and words are not thread counts; the error names the value.
    #[test]
    fn parse_threads_rejects() {
        for bad in ["0", "-1", "abc"] {
            let err = parse_threads(bad).unwrap_err();
            assert!(err.contains(bad), "{err}");
        }
        assert_eq!(parse_threads("1"), Ok(1));
        assert_eq!(parse_threads("16"), Ok(16));
    }

    /// The default thread count takes a positive performance-core count (capped at the
    /// available parallelism) and falls back to the available parallelism when the read
    /// fails or reports 0.
    #[test]
    fn default_threads_falls_back() {
        let available = available_threads();
        let missing = || Err(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert_eq!(default_threads_from(missing), available);
        assert_eq!(default_threads_from(|| Ok(0)), available);
        assert_eq!(default_threads_from(|| Ok(1)), 1);
        assert_eq!(default_threads_from(|| Ok(available + 1)), available);
        #[cfg(not(target_vendor = "apple"))]
        assert_eq!(default_threads(), available);
    }

    /// On Apple Silicon the sysctl reports at least one performance core, no more than the
    /// logical cores, and that count is the default.
    #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
    #[test]
    fn performance_cores_read_on_apple_silicon() {
        let n = performance_cores().expect("hw.perflevel0.physicalcpu");
        assert!(n >= 1 && n <= available_threads(), "{n} performance cores");
        assert_eq!(default_threads(), n);
    }

    /// Re-runs this test binary with `name` (an ignored helper test) and one environment
    /// variable set, and returns whether it succeeded and what it wrote to stderr.
    ///
    /// # Panics
    ///
    /// Panics if the child did not run the helper: a filter that matches no test also exits
    /// successfully, so the child's summary must report the one test passed or failed.
    fn run_child(name: &str, var: &str, value: &str) -> (bool, String) {
        let out = crate::test_child::run_alone_without(
            name,
            &[(var, value)],
            &["CAVITATION_THREADS", "CAVITATION_KERNEL"],
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let ok = out.status.success();
        let summary = if ok { "1 passed" } else { "1 failed" };
        assert!(
            stdout.contains(summary),
            "the child did not run `{name}`; stdout:\n{stdout}"
        );
        (ok, String::from_utf8_lossy(&out.stderr).into_owned())
    }

    /// `CAVITATION_THREADS=abc` makes `threads()` panic, naming the variable and the value.
    #[test]
    fn env_invalid_panics_in_child() {
        let (ok, stderr) = run_child(
            "kernels::tests::threads_from_env",
            "CAVITATION_THREADS",
            "abc",
        );
        assert!(!ok, "the child should fail; stderr:\n{stderr}");
        assert!(
            stderr.contains("CAVITATION_THREADS") && stderr.contains("`abc`"),
            "stderr:\n{stderr}"
        );
        let (ok, stderr) = run_child(
            "kernels::tests::threads_from_env",
            "CAVITATION_THREADS",
            "3",
        );
        assert!(ok, "stderr:\n{stderr}");
    }

    /// `CAVITATION_KERNEL` naming an unknown level, or one this CPU cannot run, makes
    /// `level()` panic with a message naming the level and the detected features.
    #[test]
    fn env_bad_kernel_panics_in_child() {
        let (ok, stderr) = run_child(
            "kernels::tests::level_from_env",
            "CAVITATION_KERNEL",
            "avx9",
        );
        assert!(!ok, "stderr:\n{stderr}");
        assert!(
            stderr.contains("CAVITATION_KERNEL") && stderr.contains("`avx9`"),
            "stderr:\n{stderr}"
        );
        let foreign = if cfg!(target_arch = "aarch64") {
            "avx2"
        } else {
            "neon"
        };
        let (ok, stderr) = run_child(
            "kernels::tests::level_from_env",
            "CAVITATION_KERNEL",
            foreign,
        );
        assert!(!ok, "stderr:\n{stderr}");
        assert!(
            stderr.contains(foreign) && stderr.contains("detected features"),
            "stderr:\n{stderr}"
        );
        let (ok, stderr) = run_child(
            "kernels::tests::level_from_env",
            "CAVITATION_KERNEL",
            "reference",
        );
        assert!(ok, "stderr:\n{stderr}");
        // `accelerate` runs on Apple Silicon and is refused, the same way, everywhere else.
        let (ok, stderr) = run_child(
            "kernels::tests::level_from_env",
            "CAVITATION_KERNEL",
            "accelerate",
        );
        if supported(Level::Accelerate) {
            assert!(ok, "stderr:\n{stderr}");
        } else {
            assert!(!ok, "stderr:\n{stderr}");
            assert!(
                stderr.contains("cannot run kernel level accelerate")
                    && stderr.contains("detected features"),
                "stderr:\n{stderr}"
            );
        }
    }

    /// Helper for `env_invalid_panics_in_child`: reads the thread count from the
    /// environment. Ignored by default; it is run in a child process.
    #[test]
    #[ignore = "run in a child process by env_invalid_panics_in_child"]
    fn threads_from_env() {
        assert!(threads() >= 1);
    }

    /// Helper for `env_bad_kernel_panics_in_child`: reads the kernel level from the
    /// environment. Ignored by default; it is run in a child process.
    #[test]
    #[ignore = "run in a child process by env_bad_kernel_panics_in_child"]
    fn level_from_env() {
        assert!(supported(level()));
    }
}
