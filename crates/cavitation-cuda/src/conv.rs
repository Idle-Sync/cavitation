//! The 2-D convolution through cuDNN: the forward pass, the input gradient, the weight
//! gradient and the bias gradient, with deterministic algorithms only.

use std::ffi::{c_int, c_void};
use std::fmt;
use std::sync::{Mutex, PoisonError};

use crate::buffer::DeviceBuffer;
use crate::device::{Bound, Cuda};
use crate::error::CudaError;
use crate::sys::{
    AlgoPerf, CUDNN_CROSS_CORRELATION, CUDNN_DETERMINISTIC, CUDNN_FLOAT, CUDNN_FMA_MATH,
    CUDNN_NCHW, CuDevice,
};

/// The most scratch memory one pass of a [`Conv2d`] may use: an algorithm that needs more
/// is passed over for the next one cuDNN ranks.
pub const MAX_WORKSPACE: usize = 256 << 20;

/// cuDNN's forward algorithms, by their number.
const FORWARD: [&str; 8] = [
    "IMPLICIT_GEMM",
    "IMPLICIT_PRECOMP_GEMM",
    "GEMM",
    "DIRECT",
    "FFT",
    "FFT_TILING",
    "WINOGRAD",
    "WINOGRAD_NONFUSED",
];
/// cuDNN's input-gradient (backward data) algorithms, by their number.
const INPUT_GRAD: [&str; 6] = [
    "ALGO_0",
    "ALGO_1",
    "FFT",
    "FFT_TILING",
    "WINOGRAD",
    "WINOGRAD_NONFUSED",
];
/// The forward algorithms never run: `WINOGRAD_NONFUSED` was measured 137 u · Σ|terms| from
/// the exact result on the 3 → 32 convolution at 32 × 32 over 128 images, above the forward
/// tolerance of `tests/conv.rs` (128), where every other forward algorithm is within 11.
const FORWARD_EXCLUDED: [&str; 1] = ["WINOGRAD_NONFUSED"];
/// The input-gradient algorithms never run: measured on the shapes of `tests/conv.rs`,
/// `FFT` up to 14 u · Σ|terms| from the exact result, `FFT_TILING` 113 and
/// `WINOGRAD_NONFUSED` 238, above the input gradient's tolerance there (8), where `ALGO_1` and
/// `WINOGRAD` are within 5.2.
const INPUT_GRAD_EXCLUDED: [&str; 3] = ["FFT", "FFT_TILING", "WINOGRAD_NONFUSED"];
/// cuDNN's weight-gradient (backward filter) algorithms, by their number.
const WEIGHT_GRAD: [&str; 7] = [
    "ALGO_0",
    "ALGO_1",
    "FFT",
    "ALGO_3",
    "WINOGRAD",
    "WINOGRAD_NONFUSED",
    "FFT_TILING",
];

/// The shape of a 2-D convolution over a batch of NCHW images, with OIHW weights, as the
/// CPU's `Conv2dGeom` describes one.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::ConvShape;
///
/// let shape = ConvShape {
///     padding: (1, 1),
///     ..ConvShape::new(128, 3, 32, (32, 32), (3, 3))
/// };
/// assert_eq!(shape.output(), Some((32, 32)));
/// assert_eq!(shape.y_len(), 128 * 32 * 32 * 32);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvShape {
    /// Images in the batch.
    pub batch: u32,
    /// Channels of each input image.
    pub in_channels: u32,
    /// Channels of each output image: the number of kernels.
    pub out_channels: u32,
    /// Height and width of each input image.
    pub input: (u32, u32),
    /// Kernel height and width.
    pub kernel: (u32, u32),
    /// Step between windows, down and across.
    pub stride: (u32, u32),
    /// Zeros added on each side, top and bottom, left and right.
    pub padding: (u32, u32),
    /// Spacing between the kernel's taps, down and across.
    pub dilation: (u32, u32),
}

impl ConvShape {
    /// A convolution with stride 1, no padding and no dilation.
    pub fn new(
        batch: u32,
        in_channels: u32,
        out_channels: u32,
        input: (u32, u32),
        kernel: (u32, u32),
    ) -> ConvShape {
        ConvShape {
            batch,
            in_channels,
            out_channels,
            input,
            kernel,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }

    /// The output's height and width, or `None` when no window fits or a size is zero.
    pub fn output(&self) -> Option<(u32, u32)> {
        let axis = |size: u32, kernel: u32, stride: u32, padding: u32, dilation: u32| {
            if kernel == 0 || stride == 0 || dilation == 0 {
                return None;
            }
            let span = u64::from(dilation) * u64::from(kernel - 1) + 1;
            let room = (u64::from(size) + 2 * u64::from(padding)).checked_sub(span)?;
            u32::try_from(room / u64::from(stride) + 1).ok()
        };
        Some((
            axis(
                self.input.0,
                self.kernel.0,
                self.stride.0,
                self.padding.0,
                self.dilation.0,
            )?,
            axis(
                self.input.1,
                self.kernel.1,
                self.stride.1,
                self.padding.1,
                self.dilation.1,
            )?,
        ))
    }

    /// The product of four sizes, or `None` if it does not fit a `usize`.
    fn product(sizes: [u32; 4]) -> Option<usize> {
        sizes
            .iter()
            .try_fold(1usize, |acc, &s| acc.checked_mul(s as usize))
    }

    /// The input's length in floats: batch × in channels × height × width, saturating at
    /// `usize::MAX` ([`Conv2d::new`] refuses a shape whose lengths do not fit).
    pub fn x_len(&self) -> usize {
        Self::product([self.batch, self.in_channels, self.input.0, self.input.1])
            .unwrap_or(usize::MAX)
    }

    /// The weight's length in floats: out channels × in channels × kernel height × width,
    /// saturating at `usize::MAX`.
    pub fn w_len(&self) -> usize {
        Self::product([
            self.out_channels,
            self.in_channels,
            self.kernel.0,
            self.kernel.1,
        ])
        .unwrap_or(usize::MAX)
    }

    /// The output's length in floats, 0 when no window fits, saturating at `usize::MAX`.
    pub fn y_len(&self) -> usize {
        self.output().map_or(0, |(h, w)| {
            Self::product([self.batch, self.out_channels, h, w]).unwrap_or(usize::MAX)
        })
    }
}

/// The algorithm cuDNN runs for each pass of a [`Conv2d`], by cuDNN's names
/// (`CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED` is `WINOGRAD_NONFUSED`), with the scratch
/// memory each needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConvAlgorithms {
    /// The forward pass's algorithm and its scratch bytes.
    pub forward: (&'static str, usize),
    /// The input gradient's algorithm and its scratch bytes.
    pub input_grad: (&'static str, usize),
    /// The weight gradient's algorithm and its scratch bytes.
    pub weight_grad: (&'static str, usize),
}

impl fmt::Display for ConvAlgorithms {
    /// Writes the three algorithms on one line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "forward {}, input gradient {}, weight gradient {}",
            self.forward.0, self.input_grad.0, self.weight_grad.0
        )
    }
}

/// A convolution's cuDNN descriptors, each null until it is created, destroyed together
/// when dropped, with the lock taken for it. They are created with the lock held, by a
/// caller that declares them before its guard, so that on an early return the guard is
/// given back before they are dropped.
struct Descriptors<'c> {
    /// The CUDA whose cuDNN made them.
    cuda: &'c Cuda,
    /// The input's tensor descriptor (and the input gradient's).
    x: *mut c_void,
    /// The weight's filter descriptor (and the weight gradient's).
    w: *mut c_void,
    /// The output's tensor descriptor (and the output gradient's).
    y: *mut c_void,
    /// The bias's tensor descriptor, 1 × out channels × 1 × 1.
    bias: *mut c_void,
    /// The convolution's descriptor.
    conv: *mut c_void,
}

impl Drop for Descriptors<'_> {
    fn drop(&mut self) {
        let api = &self.cuda.cudnn;
        let all = [
            (self.x, api.cudnnDestroyTensorDescriptor),
            (self.y, api.cudnnDestroyTensorDescriptor),
            (self.bias, api.cudnnDestroyTensorDescriptor),
            (self.w, api.cudnnDestroyFilterDescriptor),
            (self.conv, api.cudnnDestroyConvolutionDescriptor),
        ];
        if all.iter().all(|(raw, _)| raw.is_null()) {
            return;
        }
        let _bound = self.cuda.bind_anyway();
        for (raw, destroy) in all {
            if !raw.is_null() {
                // SAFETY: the descriptor came from the matching create function and is
                // destroyed once, with the lock held.
                unsafe { destroy(raw) };
            }
        }
    }
}

impl<'c> Descriptors<'c> {
    /// None created yet.
    fn new(cuda: &'c Cuda) -> Descriptors<'c> {
        let null = std::ptr::null_mut();
        Descriptors {
            cuda,
            x: null,
            w: null,
            y: null,
            bias: null,
            conv: null,
        }
    }

    /// Creates a descriptor with `create`, with the lock held.
    fn create(
        &self,
        _bound: &Bound<'_>,
        what: &'static str,
        create: unsafe extern "system" fn(*mut *mut c_void) -> c_int,
    ) -> Result<*mut c_void, CudaError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: the pointer is to a live descriptor handle, written once; the lock is
        // held.
        let status = unsafe { create(&mut raw) };
        self.check(what, status)?;
        Ok(raw)
    }

    /// Creates a 4-D float NCHW tensor descriptor, with the lock held. The caller stores
    /// it in its field before setting it, so that it is destroyed if setting fails.
    fn tensor(&self, bound: &Bound<'_>) -> Result<*mut c_void, CudaError> {
        self.create(
            bound,
            "cudnnCreateTensorDescriptor",
            self.cuda.cudnn.cudnnCreateTensorDescriptor,
        )
    }

    /// Sets the tensor descriptor `raw`, one of these, to `dims`, with the lock held.
    fn set_tensor(
        &self,
        _bound: &Bound<'_>,
        raw: *mut c_void,
        dims: [c_int; 4],
    ) -> Result<(), CudaError> {
        let [n, c, h, w] = dims;
        // SAFETY: the descriptor is live, the layout and type are cuDNN's constants, and the
        // lock is held.
        let status = unsafe {
            (self.cuda.cudnn.cudnnSetTensor4dDescriptor)(raw, CUDNN_NCHW, CUDNN_FLOAT, n, c, h, w)
        };
        self.check("cudnnSetTensor4dDescriptor", status)
    }

    /// `Ok` for a cuDNN status of 0.
    fn check(&self, what: &'static str, status: c_int) -> Result<(), CudaError> {
        if status == 0 {
            Ok(())
        } else {
            Err(self.cuda.cudnn_error(what, status))
        }
    }
}

/// One 2-D convolution shape on the card, through cuDNN: the forward pass with an optional
/// bias, the input gradient, the weight gradient and the bias gradient, in `f32`.
///
/// The algorithms are chosen once, when the convolution is made, among those that support
/// the shape, give the same bits on every run (cuDNN's `CUDNN_DETERMINISTIC`), run without
/// tensor cores (`CUDNN_FMA_MATH`, so no product is rounded to a narrower type), need at most
/// [`MAX_WORKSPACE`] bytes of scratch memory, which the convolution holds, are not Winograd
/// unless the kernel is 3 × 3, and are not among the few measured too far from the exact
/// result: the forward pass's `WINOGRAD_NONFUSED`, and the input gradient's `FFT`,
/// `FFT_TILING` and `WINOGRAD_NONFUSED`. [`Conv2d::candidates`] lists the rest. By default
/// each pass runs the first of them cuDNN's heuristics rank, so two runs of a program on one
/// card give the same bits; with [`Cuda::set_cudnn_benchmark`] (or
/// `CAVITATION_CUDNN_BENCHMARK=1`) they are timed instead (see [`Conv2d::new`]).
///
/// On a 5 × 5 kernel cuDNN's Winograd weight gradient was measured 2,782 u · Σ|terms| from
/// the exact result, where every algorithm allowed is within 18. cuDNN publishes no error
/// bound for its algorithms, so how far each is from the exact result is measured, not
/// proven: see this crate's `tests/conv.rs`, which runs every allowed algorithm of every
/// pass on its shapes.
///
/// Every pass is queued on the card and returns without waiting for it, like
/// [`Kernel::launch`](crate::Kernel::launch).
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Conv2d, ConvShape, Cuda};
///
/// if let Some(cuda) = Cuda::new() {
///     // One 2 × 2 image, one channel, a 1 × 1 kernel of 3 and a bias of 1.
///     let conv = Conv2d::new(&cuda, ConvShape::new(1, 1, 1, (2, 2), (1, 1))).unwrap();
///     let mut x = cuda.buffer(4 * 4).unwrap();
///     let mut w = cuda.buffer(4).unwrap();
///     let mut b = cuda.buffer(4).unwrap();
///     let mut y = cuda.buffer(4 * 4).unwrap();
///     x.write(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
///     w.write(&[3.0f32]).unwrap();
///     b.write(&[1.0f32]).unwrap();
///     conv.forward(&x, &w, Some(&b), &mut y).unwrap();
///     let mut out = [0.0f32; 4];
///     y.read(&mut out).unwrap();
///     assert_eq!(out, [4.0, 7.0, 10.0, 13.0]);
/// }
/// ```
pub struct Conv2d<'c> {
    /// The CUDA it runs on.
    cuda: &'c Cuda,
    /// Its shape.
    shape: ConvShape,
    /// Its cuDNN descriptors.
    d: Descriptors<'c>,
    /// The forward algorithm's number.
    forward_algo: c_int,
    /// The input-gradient algorithm's number.
    input_grad_algo: c_int,
    /// The weight-gradient algorithm's number.
    weight_grad_algo: c_int,
    /// The algorithms by name, with their scratch bytes.
    algorithms: ConvAlgorithms,
    /// Scratch memory for the pass that needs most.
    workspace: DeviceBuffer<'c>,
}

impl fmt::Debug for Conv2d<'_> {
    /// Writes the shape and the algorithms.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conv2d")
            .field("shape", &self.shape)
            .field("algorithms", &self.algorithms)
            .finish()
    }
}

/// `1.0` and `0.0` as cuDNN's scaling factors for `f32` data take them, by pointer.
const ONE: f32 = 1.0;
/// See [`ONE`].
const ZERO: f32 = 0.0;

/// A pointer to a scaling factor.
fn factor(value: &f32) -> *const c_void {
    std::ptr::from_ref(value).cast()
}

/// One algorithm a pass may run: its number in its pass's enumeration, its name and the
/// scratch bytes it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Algo {
    /// Its number.
    num: c_int,
    /// Its name, as cuDNN's enumeration has it without the prefix.
    name: &'static str,
    /// The scratch memory it needs, in bytes.
    bytes: usize,
}

/// Every algorithm of `ranked`, in its order, that supports the shape, is deterministic, runs
/// without tensor cores, is not one of `excluded` nor Winograd unless `winograd`, and needs
/// at most
/// [`MAX_WORKSPACE`] bytes of scratch memory, as `workspace` finds them; an error naming the
/// ones passed over when there is none.
fn qualifying(
    pass: &str,
    names: &[&'static str],
    excluded: &[&str],
    ranked: &[AlgoPerf],
    winograd: bool,
    mut workspace: impl FnMut(c_int) -> Result<usize, CudaError>,
) -> Result<Vec<Algo>, CudaError> {
    let mut seen = Vec::new();
    let mut found = Vec::new();
    for perf in ranked {
        let name = usize::try_from(perf.algo)
            .ok()
            .and_then(|i| names.get(i).copied())
            .unwrap_or("an unknown algorithm");
        if perf.status != 0 {
            continue;
        }
        if perf.determinism != CUDNN_DETERMINISTIC
            || perf.math_type != CUDNN_FMA_MATH
            || excluded.contains(&name)
            || (!winograd && name.starts_with("WINOGRAD"))
        {
            seen.push(name);
            continue;
        }
        let bytes = workspace(perf.algo)?;
        if bytes > MAX_WORKSPACE {
            seen.push(name);
            continue;
        }
        found.push(Algo {
            num: perf.algo,
            name,
            bytes,
        });
    }
    if found.is_empty() {
        return Err(CudaError::Invalid(format!(
            "cuDNN has no deterministic {pass} algorithm without tensor cores, within \
             {MAX_WORKSPACE} bytes of scratch memory and, for a kernel other than 3 × 3, other \
             than Winograd, for this convolution; it offered {}",
            if seen.is_empty() {
                "none".to_string()
            } else {
                seen.join(", ")
            }
        )));
    }
    Ok(found)
}

/// The algorithm of `candidates` that `timed` (cuDNN's Find, fastest first) ranks first among
/// those it ran without error and calls deterministic; the first candidate, the heuristics'
/// choice, when it ranks none.
fn fastest(candidates: &[Algo], timed: &[AlgoPerf]) -> Algo {
    timed
        .iter()
        .filter(|perf| perf.status == 0 && perf.determinism == CUDNN_DETERMINISTIC)
        .find_map(|perf| candidates.iter().find(|a| a.num == perf.algo))
        .copied()
        .unwrap_or(candidates[0])
}

/// The algorithms each pass of a convolution may run on this card, in the order cuDNN's
/// heuristics rank them, each with the scratch bytes it needs: the ones that meet the
/// conditions in [`Conv2d`]'s description. [`Conv2d::new`] runs the first of each list, or
/// the fastest when [`Cuda::set_cudnn_benchmark`] is on; [`Conv2d::with_algorithms`] runs
/// any of them.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Conv2d, ConvShape, Cuda};
///
/// if let Some(cuda) = Cuda::new() {
///     let shape = ConvShape { padding: (1, 1), ..ConvShape::new(8, 3, 4, (8, 8), (3, 3)) };
///     let allowed = Conv2d::candidates(&cuda, shape).unwrap();
///     let chosen = Conv2d::new(&cuda, shape).unwrap().algorithms();
///     assert!(allowed.forward.contains(&chosen.forward));
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConvCandidates {
    /// The forward algorithms and their scratch bytes.
    pub forward: Vec<(&'static str, usize)>,
    /// The input-gradient algorithms and their scratch bytes.
    pub input_grad: Vec<(&'static str, usize)>,
    /// The weight-gradient algorithms and their scratch bytes.
    pub weight_grad: Vec<(&'static str, usize)>,
}

/// A convolution's descriptors and the algorithms each pass may run.
struct Setup<'c> {
    /// The descriptors.
    d: Descriptors<'c>,
    /// The forward algorithms allowed, in the heuristics' order.
    forward: Vec<Algo>,
    /// The input-gradient algorithms allowed.
    input_grad: Vec<Algo>,
    /// The weight-gradient algorithms allowed.
    weight_grad: Vec<Algo>,
}

/// The algorithms timing chose in this process for each card and shape, so that every
/// [`Conv2d::new`] of one shape on one card with timing on runs the same ones, and so gives
/// the same bits, whatever the timing of a later set-up would say.
static CHOSEN: Mutex<Vec<Chosen>> = Mutex::new(Vec::new());

/// One entry of [`CHOSEN`]: the card and the shape, and the forward, input-gradient and
/// weight-gradient algorithms.
type Chosen = ((CuDevice, ConvShape), [Algo; 3]);

impl<'c> Conv2d<'c> {
    /// Sets up the convolution `shape` on `cuda`'s card and chooses its algorithms.
    ///
    /// By default each pass runs the first allowed algorithm cuDNN's heuristics rank. With
    /// [`Cuda::set_cudnn_benchmark`] on (or `CAVITATION_CUDNN_BENCHMARK=1` when `cuda` was
    /// opened), the first convolution of a shape on a card in a process times every allowed
    /// algorithm of each pass on scratch buffers (cuDNN's `cudnnFindConvolution*AlgorithmEx`)
    /// and keeps the fastest; every later one of that shape in the process runs the same
    /// ones without timing, so its runs give the same bits. Timing may choose differently in
    /// another process or on another card, which may then differ in the last bits; each
    /// choice is one of [`Conv2d::candidates`], within the measured tolerances of this
    /// crate's `tests/conv.rs`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if no window fits, a size does not fit cuDNN's `int`, or a
    /// pass has no algorithm meeting the conditions in the type's description;
    /// [`CudaError::Call`] if cuDNN refuses the shape or the scratch memory cannot be
    /// allocated.
    pub fn new(cuda: &'c Cuda, shape: ConvShape) -> Result<Conv2d<'c>, CudaError> {
        let setup = Self::set_up(cuda, shape)?;
        if !cuda.cudnn_benchmark() {
            let algos = [setup.forward[0], setup.input_grad[0], setup.weight_grad[0]];
            return Self::build(cuda, shape, setup, algos);
        }
        let key = (cuda.device(), shape);
        let mut chosen = CHOSEN.lock().unwrap_or_else(PoisonError::into_inner);
        let algos = match chosen.iter().find(|(k, _)| *k == key) {
            Some(&(_, algos)) => algos,
            None => {
                let algos = Self::time(cuda, shape, &setup)?;
                chosen.push((key, algos));
                algos
            }
        };
        drop(chosen);
        Self::build(cuda, shape, setup, algos)
    }

    /// The algorithms each pass of `shape` may run on `cuda`'s card.
    ///
    /// # Errors
    ///
    /// As [`Conv2d::new`].
    pub fn candidates(cuda: &Cuda, shape: ConvShape) -> Result<ConvCandidates, CudaError> {
        let setup = Self::set_up(cuda, shape)?;
        let list = |v: &[Algo]| v.iter().map(|a| (a.name, a.bytes)).collect();
        Ok(ConvCandidates {
            forward: list(&setup.forward),
            input_grad: list(&setup.input_grad),
            weight_grad: list(&setup.weight_grad),
        })
    }

    /// Sets up the convolution `shape` on `cuda`'s card with the named algorithms, each one
    /// of [`Conv2d::candidates`] for its pass, without timing.
    ///
    /// # Errors
    ///
    /// As [`Conv2d::new`], and [`CudaError::Invalid`] if a name is not one of its pass's
    /// candidates.
    pub fn with_algorithms(
        cuda: &'c Cuda,
        shape: ConvShape,
        forward: &str,
        input_grad: &str,
        weight_grad: &str,
    ) -> Result<Conv2d<'c>, CudaError> {
        let setup = Self::set_up(cuda, shape)?;
        let named = |pass: &str, list: &[Algo], name: &str| {
            list.iter()
                .find(|a| a.name == name)
                .copied()
                .ok_or_else(|| {
                    CudaError::Invalid(format!(
                        "convolution {shape:?}: {name} is not an allowed {pass} algorithm"
                    ))
                })
        };
        let algos = [
            named("forward", &setup.forward, forward)?,
            named("input-gradient", &setup.input_grad, input_grad)?,
            named("weight-gradient", &setup.weight_grad, weight_grad)?,
        ];
        Self::build(cuda, shape, setup, algos)
    }

    /// The convolution with its descriptors and the algorithms `algos` (forward, input
    /// gradient, weight gradient), and scratch memory for the one that needs most.
    fn build(
        cuda: &'c Cuda,
        shape: ConvShape,
        setup: Setup<'c>,
        algos: [Algo; 3],
    ) -> Result<Conv2d<'c>, CudaError> {
        let [forward, input_grad, weight_grad] = algos;
        let workspace = cuda.buffer(forward.bytes.max(input_grad.bytes).max(weight_grad.bytes))?;
        Ok(Conv2d {
            cuda,
            shape,
            d: setup.d,
            forward_algo: forward.num,
            input_grad_algo: input_grad.num,
            weight_grad_algo: weight_grad.num,
            algorithms: ConvAlgorithms {
                forward: (forward.name, forward.bytes),
                input_grad: (input_grad.name, input_grad.bytes),
                weight_grad: (weight_grad.name, weight_grad.bytes),
            },
            workspace,
        })
    }

    /// Times every allowed algorithm of each pass of `shape` on zeroed scratch buffers with
    /// cuDNN's Find, and returns the fastest of each.
    fn time(cuda: &'c Cuda, shape: ConvShape, setup: &Setup<'c>) -> Result<[Algo; 3], CudaError> {
        let all = || {
            (setup.forward.iter())
                .chain(&setup.input_grad)
                .chain(&setup.weight_grad)
        };
        let bytes = all().map(|a| a.bytes).max().unwrap_or(0);
        // Declared before the guard, so that they are freed after it is given back. The
        // input gradient is written over the input, and the weight gradient over the weight:
        // their values are not read.
        let x = cuda.buffer(4 * shape.x_len())?;
        let w = cuda.buffer(4 * shape.w_len())?;
        let y = cuda.buffer(4 * shape.y_len())?;
        let scratch = cuda.buffer(bytes)?;
        let d = &setup.d;
        let handle = cuda.cudnn_handle;
        let api = &cuda.cudnn;
        let mut timed = [AlgoPerf::default(); 16];
        let count = timed.len() as c_int;
        let mut returned: c_int = 0;
        let list = |status: c_int, what, returned: c_int, timed: &[AlgoPerf]| {
            d.check(what, status)?;
            Ok::<_, CudaError>(
                timed[..usize::try_from(returned).unwrap_or(0).min(timed.len())].to_vec(),
            )
        };
        let bound = cuda.bind()?;
        // SAFETY: the handle and descriptors are live; each buffer holds its tensor and the
        // scratch buffer `bytes` bytes; `timed` holds `count` results; the lock is held, as
        // for the two calls below.
        let status = unsafe {
            (api.cudnnFindConvolutionForwardAlgorithmEx)(
                handle,
                d.x,
                x.pointer,
                d.w,
                w.pointer,
                d.conv,
                d.y,
                y.pointer,
                count,
                &mut returned,
                timed.as_mut_ptr(),
                scratch.pointer,
                bytes,
            )
        };
        let found = list(
            status,
            "cudnnFindConvolutionForwardAlgorithmEx",
            returned,
            &timed,
        )?;
        let forward = fastest(&setup.forward, &found);
        // SAFETY: as above; dx is written over x.
        let status = unsafe {
            (api.cudnnFindConvolutionBackwardDataAlgorithmEx)(
                handle,
                d.w,
                w.pointer,
                d.y,
                y.pointer,
                d.conv,
                d.x,
                x.pointer,
                count,
                &mut returned,
                timed.as_mut_ptr(),
                scratch.pointer,
                bytes,
            )
        };
        let found = list(
            status,
            "cudnnFindConvolutionBackwardDataAlgorithmEx",
            returned,
            &timed,
        )?;
        let input_grad = fastest(&setup.input_grad, &found);
        // SAFETY: as above; dw is written over w.
        let status = unsafe {
            (api.cudnnFindConvolutionBackwardFilterAlgorithmEx)(
                handle,
                d.x,
                x.pointer,
                d.y,
                y.pointer,
                d.conv,
                d.w,
                w.pointer,
                count,
                &mut returned,
                timed.as_mut_ptr(),
                scratch.pointer,
                bytes,
            )
        };
        let found = list(
            status,
            "cudnnFindConvolutionBackwardFilterAlgorithmEx",
            returned,
            &timed,
        )?;
        let weight_grad = fastest(&setup.weight_grad, &found);
        drop(bound);
        Ok([forward, input_grad, weight_grad])
    }

    /// Makes the descriptors of `shape` on `cuda`'s card and lists the algorithms each pass
    /// may run.
    fn set_up<'a>(cuda: &'a Cuda, shape: ConvShape) -> Result<Setup<'a>, CudaError> {
        let invalid = |why: &str| CudaError::Invalid(format!("convolution {shape:?}: {why}"));
        let (oh, ow) = shape
            .output()
            .ok_or_else(|| invalid("no window fits the input, or a size is zero"))?;
        if shape.batch == 0 || shape.in_channels == 0 || shape.out_channels == 0 {
            return Err(invalid(
                "the batch and both channel counts must be positive",
            ));
        }
        let int = |v: u32| c_int::try_from(v).map_err(|_| invalid("a size is too large for cuDNN"));
        let lens = [shape.x_len(), shape.w_len(), shape.y_len()];
        if lens.iter().any(|&n| c_int::try_from(n).is_err()) {
            return Err(invalid("a tensor has too many elements for cuDNN"));
        }
        let (oh, ow) = (int(oh)?, int(ow)?);
        let (kh, kw) = (int(shape.kernel.0)?, int(shape.kernel.1)?);
        let (n, cin, cout) = (
            int(shape.batch)?,
            int(shape.in_channels)?,
            int(shape.out_channels)?,
        );
        let (ih, iw) = (int(shape.input.0)?, int(shape.input.1)?);
        let padding = (int(shape.padding.0)?, int(shape.padding.1)?);
        let stride = (int(shape.stride.0)?, int(shape.stride.1)?);
        let dilation = (int(shape.dilation.0)?, int(shape.dilation.1)?);

        // Declared before the guard, so that on an early return the lock is given back
        // before they are destroyed, which takes it again.
        let mut d = Descriptors::new(cuda);
        let bound = cuda.bind()?;
        let api = &cuda.cudnn;
        d.x = d.tensor(&bound)?;
        d.set_tensor(&bound, d.x, [n, cin, ih, iw])?;
        d.y = d.tensor(&bound)?;
        d.set_tensor(&bound, d.y, [n, cout, oh, ow])?;
        d.bias = d.tensor(&bound)?;
        d.set_tensor(&bound, d.bias, [1, cout, 1, 1])?;
        d.w = d.create(
            &bound,
            "cudnnCreateFilterDescriptor",
            api.cudnnCreateFilterDescriptor,
        )?;
        // SAFETY: the descriptor is live; the type and layout are cuDNN's constants; the
        // lock is held, as for every cuDNN call below.
        let status = unsafe {
            (api.cudnnSetFilter4dDescriptor)(d.w, CUDNN_FLOAT, CUDNN_NCHW, cout, cin, kh, kw)
        };
        d.check("cudnnSetFilter4dDescriptor", status)?;
        d.conv = d.create(
            &bound,
            "cudnnCreateConvolutionDescriptor",
            api.cudnnCreateConvolutionDescriptor,
        )?;
        // SAFETY: the descriptor is live; the mode and type are cuDNN's constants.
        let status = unsafe {
            (api.cudnnSetConvolution2dDescriptor)(
                d.conv,
                padding.0,
                padding.1,
                stride.0,
                stride.1,
                dilation.0,
                dilation.1,
                CUDNN_CROSS_CORRELATION,
                CUDNN_FLOAT,
            )
        };
        d.check("cudnnSetConvolution2dDescriptor", status)?;
        // SAFETY: the descriptor is live, and the math type is cuDNN's constant.
        let status = unsafe { (api.cudnnSetConvolutionMathType)(d.conv, CUDNN_FMA_MATH) };
        d.check("cudnnSetConvolutionMathType", status)?;

        let mut dims: [c_int; 4] = [0; 4];
        let [dn, dc, dh, dw] = &mut dims;
        // SAFETY: the descriptors are live, and each pointer is to a live `c_int`.
        let status = unsafe {
            (api.cudnnGetConvolution2dForwardOutputDim)(d.conv, d.x, d.w, dn, dc, dh, dw)
        };
        d.check("cudnnGetConvolution2dForwardOutputDim", status)?;
        if dims != [n, cout, oh, ow] {
            return Err(invalid(&format!(
                "cuDNN's output is {dims:?}, not [{n}, {cout}, {oh}, {ow}]"
            )));
        }

        // Winograd's transforms lose accuracy as the kernel grows; it is used for 3 × 3 only.
        let winograd = shape.kernel == (3, 3);
        let handle = cuda.cudnn_handle;
        let mut ranked = [AlgoPerf::default(); 16];
        let mut returned: c_int = 0;
        let count = ranked.len() as c_int;
        let rank = |status: c_int, what, returned: c_int, ranked: &[AlgoPerf]| {
            d.check(what, status)?;
            Ok::<_, CudaError>(
                ranked[..usize::try_from(returned).unwrap_or(0).min(ranked.len())].to_vec(),
            )
        };

        // SAFETY: the handle and descriptors are live, and `ranked` holds `count` results.
        let status = unsafe {
            (api.cudnnGetConvolutionForwardAlgorithm_v7)(
                handle,
                d.x,
                d.w,
                d.conv,
                d.y,
                count,
                &mut returned,
                ranked.as_mut_ptr(),
            )
        };
        let list = rank(
            status,
            "cudnnGetConvolutionForwardAlgorithm_v7",
            returned,
            &ranked,
        )?;
        let forward = qualifying(
            "forward",
            &FORWARD,
            &FORWARD_EXCLUDED,
            &list,
            winograd,
            |algo| {
                let mut bytes = 0usize;
                // SAFETY: as above, with a pointer to a live `usize`.
                let status = unsafe {
                    (api.cudnnGetConvolutionForwardWorkspaceSize)(
                        handle, d.x, d.w, d.conv, d.y, algo, &mut bytes,
                    )
                };
                d.check("cudnnGetConvolutionForwardWorkspaceSize", status)?;
                Ok(bytes)
            },
        )?;

        // SAFETY: as for the forward ranking.
        let status = unsafe {
            (api.cudnnGetConvolutionBackwardDataAlgorithm_v7)(
                handle,
                d.w,
                d.y,
                d.conv,
                d.x,
                count,
                &mut returned,
                ranked.as_mut_ptr(),
            )
        };
        let list = rank(
            status,
            "cudnnGetConvolutionBackwardDataAlgorithm_v7",
            returned,
            &ranked,
        )?;
        let input_grad = qualifying(
            "input-gradient",
            &INPUT_GRAD,
            &INPUT_GRAD_EXCLUDED,
            &list,
            winograd,
            |algo| {
                let mut bytes = 0usize;
                // SAFETY: as above, with a pointer to a live `usize`.
                let status = unsafe {
                    (api.cudnnGetConvolutionBackwardDataWorkspaceSize)(
                        handle, d.w, d.y, d.conv, d.x, algo, &mut bytes,
                    )
                };
                d.check("cudnnGetConvolutionBackwardDataWorkspaceSize", status)?;
                Ok(bytes)
            },
        )?;

        // SAFETY: as for the forward ranking.
        let status = unsafe {
            (api.cudnnGetConvolutionBackwardFilterAlgorithm_v7)(
                handle,
                d.x,
                d.y,
                d.conv,
                d.w,
                count,
                &mut returned,
                ranked.as_mut_ptr(),
            )
        };
        let list = rank(
            status,
            "cudnnGetConvolutionBackwardFilterAlgorithm_v7",
            returned,
            &ranked,
        )?;
        let weight_grad = qualifying(
            "weight-gradient",
            &WEIGHT_GRAD,
            &[],
            &list,
            winograd,
            |algo| {
                let mut bytes = 0usize;
                // SAFETY: as above, with a pointer to a live `usize`.
                let status = unsafe {
                    (api.cudnnGetConvolutionBackwardFilterWorkspaceSize)(
                        handle, d.x, d.y, d.conv, d.w, algo, &mut bytes,
                    )
                };
                d.check("cudnnGetConvolutionBackwardFilterWorkspaceSize", status)?;
                Ok(bytes)
            },
        )?;
        drop(bound);
        Ok(Setup {
            d,
            forward,
            input_grad,
            weight_grad,
        })
    }

    /// The convolution's shape.
    pub fn shape(&self) -> ConvShape {
        self.shape
    }

    /// The algorithm chosen for each pass.
    pub fn algorithms(&self) -> ConvAlgorithms {
        self.algorithms
    }

    /// Asserts that `buffer` holds at least `floats` floats.
    fn fits(buffer: &DeviceBuffer<'_>, floats: usize, what: &str) {
        let needs = floats.saturating_mul(4);
        assert!(
            buffer.bytes() >= needs,
            "the {what} buffer holds {} bytes, and the convolution needs {needs}",
            buffer.bytes()
        );
    }

    /// `Ok` for a cuDNN status of 0.
    fn check(&self, what: &'static str, status: c_int) -> Result<(), CudaError> {
        self.d.check(what, status)
    }

    /// `y = conv(x, w)`, plus `bias` per output channel when given: `x` NCHW, `w` OIHW,
    /// `bias` one value per output channel, `y` NCHW. Every output is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuDNN refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its tensor.
    pub fn forward(
        &self,
        x: &DeviceBuffer<'_>,
        w: &DeviceBuffer<'_>,
        bias: Option<&DeviceBuffer<'_>>,
        y: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(x, s.x_len(), "input");
        Self::fits(w, s.w_len(), "weight");
        Self::fits(y, s.y_len(), "output");
        if let Some(b) = bias {
            Self::fits(b, s.out_channels as usize, "bias");
        }
        let _bound = self.cuda.bind()?;
        let api = &self.cuda.cudnn;
        // SAFETY: the handle and descriptors are live, every buffer holds its tensor
        // (checked above), the scratch buffer holds the algorithm's bytes, and the output,
        // borrowed mutably, is none of the inputs. With beta 0 the output is not read.
        let status = unsafe {
            (api.cudnnConvolutionForward)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.d.x,
                x.pointer,
                self.d.w,
                w.pointer,
                self.d.conv,
                self.forward_algo,
                self.workspace.pointer,
                self.algorithms.forward.1,
                factor(&ZERO),
                self.d.y,
                y.pointer,
            )
        };
        self.check("cudnnConvolutionForward", status)?;
        if let Some(b) = bias {
            // SAFETY: the bias holds one value per output channel, broadcast over the
            // output, which is live and borrowed mutably.
            let status = unsafe {
                (api.cudnnAddTensor)(
                    self.cuda.cudnn_handle,
                    factor(&ONE),
                    self.d.bias,
                    b.pointer,
                    factor(&ONE),
                    self.d.y,
                    y.pointer,
                )
            };
            self.check("cudnnAddTensor", status)?;
        }
        Ok(())
    }

    /// `dx`, the gradient of the input, from the weight `w` and the output's gradient `dy`.
    /// Every value of `dx` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuDNN refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its tensor.
    pub fn input_grad(
        &self,
        w: &DeviceBuffer<'_>,
        dy: &DeviceBuffer<'_>,
        dx: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(w, s.w_len(), "weight");
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(dx, s.x_len(), "input gradient");
        let _bound = self.cuda.bind()?;
        // SAFETY: as in `forward`, for the input-gradient algorithm.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardData)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.d.w,
                w.pointer,
                self.d.y,
                dy.pointer,
                self.d.conv,
                self.input_grad_algo,
                self.workspace.pointer,
                self.algorithms.input_grad.1,
                factor(&ZERO),
                self.d.x,
                dx.pointer,
            )
        };
        self.check("cudnnConvolutionBackwardData", status)
    }

    /// `dw`, the gradient of the weight, from the input `x` and the output's gradient `dy`,
    /// summed over the batch. Every value of `dw` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuDNN refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its tensor.
    pub fn weight_grad(
        &self,
        x: &DeviceBuffer<'_>,
        dy: &DeviceBuffer<'_>,
        dw: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(x, s.x_len(), "input");
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(dw, s.w_len(), "weight gradient");
        let _bound = self.cuda.bind()?;
        // SAFETY: as in `forward`, for the weight-gradient algorithm.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardFilter)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.d.x,
                x.pointer,
                self.d.y,
                dy.pointer,
                self.d.conv,
                self.weight_grad_algo,
                self.workspace.pointer,
                self.algorithms.weight_grad.1,
                factor(&ZERO),
                self.d.w,
                dw.pointer,
            )
        };
        self.check("cudnnConvolutionBackwardFilter", status)
    }

    /// `db`, the gradient of the bias: the output's gradient `dy` summed over the batch and
    /// the positions, one value per output channel. Every value of `db` is written.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if cuDNN refuses the call.
    ///
    /// # Panics
    ///
    /// If a buffer is smaller than its tensor.
    pub fn bias_grad(
        &self,
        dy: &DeviceBuffer<'_>,
        db: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let s = &self.shape;
        Self::fits(dy, s.y_len(), "output gradient");
        Self::fits(db, s.out_channels as usize, "bias gradient");
        let _bound = self.cuda.bind()?;
        // SAFETY: the handle and descriptors are live, both buffers hold their tensors
        // (checked above), and with beta 0 the gradient is not read.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardBias)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.d.y,
                dy.pointer,
                factor(&ZERO),
                self.d.bias,
                db.pointer,
            )
        };
        self.check("cudnnConvolutionBackwardBias", status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A length too large for a `usize` saturates instead of wrapping, and is refused by
    /// `Conv2d::new` through cuDNN's `int` check.
    #[test]
    fn lengths_do_not_wrap() {
        let huge = ConvShape::new(1 << 22, 1, 1, (1 << 21, 1 << 21), (1, 1));
        assert_eq!(huge.x_len(), usize::MAX);
        assert_eq!(huge.y_len(), usize::MAX);
        assert_eq!(huge.w_len(), 1);
    }

    /// The output size follows the CPU's formula, and a window that does not fit gives
    /// none.
    #[test]
    fn output_sizes() {
        let cifar = ConvShape {
            padding: (1, 1),
            ..ConvShape::new(128, 3, 32, (32, 32), (3, 3))
        };
        assert_eq!(cifar.output(), Some((32, 32)));
        let strided = ConvShape {
            stride: (2, 1),
            padding: (1, 0),
            ..ConvShape::new(1, 1, 2, (6, 7), (3, 2))
        };
        assert_eq!(strided.output(), Some((3, 6)));
        let dilated = ConvShape {
            padding: (2, 2),
            dilation: (2, 2),
            ..ConvShape::new(1, 3, 2, (7, 7), (3, 3))
        };
        assert_eq!(dilated.output(), Some((7, 7)));
        assert_eq!(ConvShape::new(1, 1, 1, (2, 2), (3, 3)).output(), None);
        let zero_stride = ConvShape {
            stride: (0, 1),
            ..ConvShape::new(1, 1, 1, (4, 4), (1, 1))
        };
        assert_eq!(zero_stride.output(), None);
        assert_eq!(zero_stride.y_len(), 0);
    }

    /// The algorithms that are deterministic, without tensor cores and within the scratch
    /// limit are kept in their order; the error names the ones passed over.
    #[test]
    fn qualifying_passes_over_what_does_not_qualify() {
        let perf = |algo, status, determinism, math_type| AlgoPerf {
            algo,
            status,
            determinism,
            math_type,
            ..AlgoPerf::default()
        };
        let ranked = [
            perf(0, 0, 0, CUDNN_FMA_MATH),
            perf(1, 9, 1, CUDNN_FMA_MATH),
            perf(2, 0, 1, 1),
            perf(3, 0, 1, CUDNN_FMA_MATH),
            perf(5, 0, 1, CUDNN_FMA_MATH),
            perf(6, 0, 1, CUDNN_FMA_MATH),
        ];
        let big = |algo| Ok(if algo == 3 { MAX_WORKSPACE + 1 } else { 64 });
        let algo = |num, name| Algo {
            num,
            name,
            bytes: 64,
        };
        assert_eq!(
            qualifying("weight-gradient", &WEIGHT_GRAD, &[], &ranked, true, big).unwrap(),
            [algo(5, "WINOGRAD_NONFUSED"), algo(6, "FFT_TILING")]
        );
        let err = qualifying(
            "weight-gradient",
            &WEIGHT_GRAD,
            &[],
            &ranked[..4],
            true,
            big,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("it offered ALGO_0, FFT, ALGO_3"), "{err}");
        // Without Winograd, the fifth (WINOGRAD_NONFUSED) is passed over too.
        assert_eq!(
            qualifying("weight-gradient", &WEIGHT_GRAD, &[], &ranked, false, big).unwrap(),
            [algo(6, "FFT_TILING")]
        );
        let err = qualifying(
            "weight-gradient",
            &WEIGHT_GRAD,
            &[],
            &ranked[..5],
            false,
            big,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("ALGO_3, WINOGRAD_NONFUSED"), "{err}");
        // An excluded algorithm is passed over and named.
        assert_eq!(
            qualifying(
                "weight-gradient",
                &WEIGHT_GRAD,
                &["WINOGRAD_NONFUSED"],
                &ranked,
                true,
                big
            )
            .unwrap(),
            [algo(6, "FFT_TILING")]
        );
    }

    /// The fastest candidate Find ran without error and calls deterministic is chosen; one
    /// it does not rank falls back to the heuristics' first.
    #[test]
    fn the_fastest_allowed_is_chosen() {
        let candidates = [
            Algo {
                num: 1,
                name: "ALGO_1",
                bytes: 0,
            },
            Algo {
                num: 2,
                name: "FFT",
                bytes: 8,
            },
        ];
        let timed = |algo, status, determinism| AlgoPerf {
            algo,
            status,
            determinism,
            ..AlgoPerf::default()
        };
        // 0 is not a candidate, 3 failed, 2 is not deterministic here: 1 wins.
        let found = [
            timed(0, 0, 1),
            timed(3, 1, 1),
            timed(2, 0, 0),
            timed(1, 0, 1),
        ];
        assert_eq!(fastest(&candidates, &found), candidates[0]);
        let found = [timed(2, 0, 1), timed(1, 0, 1)];
        assert_eq!(fastest(&candidates, &found), candidates[1]);
        assert_eq!(fastest(&candidates, &[]), candidates[0]);
    }
}
