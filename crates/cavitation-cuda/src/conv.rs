//! The 2-D convolution through cuDNN: the forward pass, the input gradient, the weight
//! gradient and the bias gradient, with deterministic algorithms only.

use std::ffi::{c_int, c_void};
use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::sys::{
    AlgoPerf, CUDNN_CROSS_CORRELATION, CUDNN_DETERMINISTIC, CUDNN_FLOAT, CUDNN_FMA_MATH, CUDNN_NCHW,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    /// The input's length in floats: batch × in channels × height × width.
    pub fn x_len(&self) -> usize {
        self.batch as usize
            * self.in_channels as usize
            * self.input.0 as usize
            * self.input.1 as usize
    }

    /// The weight's length in floats: out channels × in channels × kernel height × width.
    pub fn w_len(&self) -> usize {
        self.out_channels as usize
            * self.in_channels as usize
            * self.kernel.0 as usize
            * self.kernel.1 as usize
    }

    /// The output's length in floats, 0 when no window fits.
    pub fn y_len(&self) -> usize {
        self.output().map_or(0, |(h, w)| {
            self.batch as usize * self.out_channels as usize * h as usize * w as usize
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

/// A cuDNN descriptor, destroyed when dropped.
struct Descriptor<'c> {
    /// The CUDA whose cuDNN made it.
    cuda: &'c Cuda,
    /// The descriptor.
    raw: *mut c_void,
    /// cuDNN's function that destroys it.
    destroy: unsafe extern "system" fn(*mut c_void) -> c_int,
}

impl Drop for Descriptor<'_> {
    fn drop(&mut self) {
        // SAFETY: the descriptor came from the matching create function and is destroyed
        // once.
        unsafe { (self.destroy)(self.raw) };
    }
}

impl<'c> Descriptor<'c> {
    /// Creates a descriptor with `create`, to be destroyed with `destroy`.
    fn new(
        cuda: &'c Cuda,
        what: &'static str,
        create: unsafe extern "system" fn(*mut *mut c_void) -> c_int,
        destroy: unsafe extern "system" fn(*mut c_void) -> c_int,
    ) -> Result<Descriptor<'c>, CudaError> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: the pointer is to a live descriptor handle, written once.
        let status = unsafe { create(&mut raw) };
        if status != 0 {
            return Err(cuda.cudnn_error(what, status));
        }
        Ok(Descriptor { cuda, raw, destroy })
    }

    /// A 4-D float NCHW tensor descriptor.
    fn tensor(cuda: &'c Cuda, dims: [c_int; 4]) -> Result<Descriptor<'c>, CudaError> {
        let api = &cuda.cudnn;
        let desc = Descriptor::new(
            cuda,
            "cudnnCreateTensorDescriptor",
            api.cudnnCreateTensorDescriptor,
            api.cudnnDestroyTensorDescriptor,
        )?;
        let [n, c, h, w] = dims;
        // SAFETY: the descriptor is live, and the layout and type are cuDNN's constants.
        let status = unsafe {
            (api.cudnnSetTensor4dDescriptor)(desc.raw, CUDNN_NCHW, CUDNN_FLOAT, n, c, h, w)
        };
        desc.check("cudnnSetTensor4dDescriptor", status)?;
        Ok(desc)
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
/// The algorithms are chosen once, when the convolution is made: for each pass the first
/// one cuDNN's heuristics rank that supports the shape, gives the same bits on every run
/// (cuDNN's `CUDNN_DETERMINISTIC`), runs without tensor cores (`CUDNN_FMA_MATH`, so no
/// product is rounded to a narrower type) and needs at most [`MAX_WORKSPACE`] bytes of
/// scratch memory, which the convolution holds. cuDNN publishes no error bound for its
/// algorithms, so how far each is from the exact result is measured, not proven: see this
/// crate's `tests/conv.rs`.
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
    /// The input's descriptor (and the input gradient's).
    x: Descriptor<'c>,
    /// The weight's descriptor (and the weight gradient's).
    w: Descriptor<'c>,
    /// The output's descriptor (and the output gradient's).
    y: Descriptor<'c>,
    /// The bias's descriptor, 1 × out channels × 1 × 1.
    bias: Descriptor<'c>,
    /// The convolution's descriptor.
    conv: Descriptor<'c>,
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

/// The first algorithm of `ranked` that supports the shape, is deterministic and runs
/// without tensor cores, among the first `returned`, with its number and its name in
/// `names`; its scratch bytes are found by `workspace`, and one needing more than
/// [`MAX_WORKSPACE`] is passed over.
fn pick(
    pass: &str,
    names: &[&'static str],
    ranked: &[AlgoPerf],
    mut workspace: impl FnMut(c_int) -> Result<usize, CudaError>,
) -> Result<(c_int, &'static str, usize), CudaError> {
    let mut seen = Vec::new();
    for perf in ranked {
        let name = usize::try_from(perf.algo)
            .ok()
            .and_then(|i| names.get(i).copied())
            .unwrap_or("an unknown algorithm");
        if perf.status != 0 {
            continue;
        }
        if perf.determinism != CUDNN_DETERMINISTIC || perf.math_type != CUDNN_FMA_MATH {
            seen.push(name);
            continue;
        }
        let bytes = workspace(perf.algo)?;
        if bytes > MAX_WORKSPACE {
            seen.push(name);
            continue;
        }
        return Ok((perf.algo, name, bytes));
    }
    Err(CudaError::Invalid(format!(
        "cuDNN has no deterministic {pass} algorithm without tensor cores and within \
         {MAX_WORKSPACE} bytes of scratch memory for this convolution; it offered {}",
        if seen.is_empty() {
            "none".to_string()
        } else {
            seen.join(", ")
        }
    )))
}

impl<'c> Conv2d<'c> {
    /// Sets up the convolution `shape` on `cuda`'s card and chooses its algorithms.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] if no window fits, a size does not fit cuDNN's `int`, or a
    /// pass has no algorithm meeting the conditions in the type's description;
    /// [`CudaError::Call`] if cuDNN refuses the shape or the scratch memory cannot be
    /// allocated.
    pub fn new(cuda: &'c Cuda, shape: ConvShape) -> Result<Conv2d<'c>, CudaError> {
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
        cuda.bind()?;
        let api = &cuda.cudnn;

        let (n, cin, cout) = (
            int(shape.batch)?,
            int(shape.in_channels)?,
            int(shape.out_channels)?,
        );
        let x = Descriptor::tensor(cuda, [n, cin, int(shape.input.0)?, int(shape.input.1)?])?;
        let y = Descriptor::tensor(cuda, [n, cout, int(oh)?, int(ow)?])?;
        let bias = Descriptor::tensor(cuda, [1, cout, 1, 1])?;
        let w = Descriptor::new(
            cuda,
            "cudnnCreateFilterDescriptor",
            api.cudnnCreateFilterDescriptor,
            api.cudnnDestroyFilterDescriptor,
        )?;
        // SAFETY: the descriptor is live; the type and layout are cuDNN's constants.
        let status = unsafe {
            (api.cudnnSetFilter4dDescriptor)(
                w.raw,
                CUDNN_FLOAT,
                CUDNN_NCHW,
                cout,
                cin,
                int(shape.kernel.0)?,
                int(shape.kernel.1)?,
            )
        };
        w.check("cudnnSetFilter4dDescriptor", status)?;
        let conv = Descriptor::new(
            cuda,
            "cudnnCreateConvolutionDescriptor",
            api.cudnnCreateConvolutionDescriptor,
            api.cudnnDestroyConvolutionDescriptor,
        )?;
        // SAFETY: the descriptor is live; the mode and type are cuDNN's constants.
        let status = unsafe {
            (api.cudnnSetConvolution2dDescriptor)(
                conv.raw,
                int(shape.padding.0)?,
                int(shape.padding.1)?,
                int(shape.stride.0)?,
                int(shape.stride.1)?,
                int(shape.dilation.0)?,
                int(shape.dilation.1)?,
                CUDNN_CROSS_CORRELATION,
                CUDNN_FLOAT,
            )
        };
        conv.check("cudnnSetConvolution2dDescriptor", status)?;
        // SAFETY: the descriptor is live, and the math type is cuDNN's constant.
        let status = unsafe { (api.cudnnSetConvolutionMathType)(conv.raw, CUDNN_FMA_MATH) };
        conv.check("cudnnSetConvolutionMathType", status)?;

        let mut dims: [c_int; 4] = [0; 4];
        let [dn, dc, dh, dw] = &mut dims;
        // SAFETY: the descriptors are live, and each pointer is to a live `c_int`.
        let status = unsafe {
            (api.cudnnGetConvolution2dForwardOutputDim)(conv.raw, x.raw, w.raw, dn, dc, dh, dw)
        };
        conv.check("cudnnGetConvolution2dForwardOutputDim", status)?;
        if dims != [n, cout, int(oh)?, int(ow)?] {
            return Err(invalid(&format!(
                "cuDNN's output is {dims:?}, not [{n}, {cout}, {oh}, {ow}]"
            )));
        }

        let handle = cuda.cudnn_handle;
        let mut ranked = [AlgoPerf::default(); 16];
        let mut returned: c_int = 0;
        let count = ranked.len() as c_int;
        let rank = |status: c_int, what, returned: c_int, ranked: &[AlgoPerf]| {
            conv.check(what, status)?;
            Ok::<_, CudaError>(
                ranked[..usize::try_from(returned).unwrap_or(0).min(ranked.len())].to_vec(),
            )
        };

        // SAFETY: the handle and descriptors are live, and `ranked` holds `count` results.
        let status = unsafe {
            (api.cudnnGetConvolutionForwardAlgorithm_v7)(
                handle,
                x.raw,
                w.raw,
                conv.raw,
                y.raw,
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
        let forward = pick("forward", &FORWARD, &list, |algo| {
            let mut bytes = 0usize;
            // SAFETY: as above, with a pointer to a live `usize`.
            let status = unsafe {
                (api.cudnnGetConvolutionForwardWorkspaceSize)(
                    handle, x.raw, w.raw, conv.raw, y.raw, algo, &mut bytes,
                )
            };
            conv.check("cudnnGetConvolutionForwardWorkspaceSize", status)?;
            Ok(bytes)
        })?;

        // SAFETY: as for the forward ranking.
        let status = unsafe {
            (api.cudnnGetConvolutionBackwardDataAlgorithm_v7)(
                handle,
                w.raw,
                y.raw,
                conv.raw,
                x.raw,
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
        let input_grad = pick("input-gradient", &INPUT_GRAD, &list, |algo| {
            let mut bytes = 0usize;
            // SAFETY: as above, with a pointer to a live `usize`.
            let status = unsafe {
                (api.cudnnGetConvolutionBackwardDataWorkspaceSize)(
                    handle, w.raw, y.raw, conv.raw, x.raw, algo, &mut bytes,
                )
            };
            conv.check("cudnnGetConvolutionBackwardDataWorkspaceSize", status)?;
            Ok(bytes)
        })?;

        // SAFETY: as for the forward ranking.
        let status = unsafe {
            (api.cudnnGetConvolutionBackwardFilterAlgorithm_v7)(
                handle,
                x.raw,
                y.raw,
                conv.raw,
                w.raw,
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
        let weight_grad = pick("weight-gradient", &WEIGHT_GRAD, &list, |algo| {
            let mut bytes = 0usize;
            // SAFETY: as above, with a pointer to a live `usize`.
            let status = unsafe {
                (api.cudnnGetConvolutionBackwardFilterWorkspaceSize)(
                    handle, x.raw, y.raw, conv.raw, w.raw, algo, &mut bytes,
                )
            };
            conv.check("cudnnGetConvolutionBackwardFilterWorkspaceSize", status)?;
            Ok(bytes)
        })?;

        let workspace = cuda.buffer(forward.2.max(input_grad.2).max(weight_grad.2))?;
        Ok(Conv2d {
            cuda,
            shape,
            x,
            w,
            y,
            bias,
            conv,
            forward_algo: forward.0,
            input_grad_algo: input_grad.0,
            weight_grad_algo: weight_grad.0,
            algorithms: ConvAlgorithms {
                forward: (forward.1, forward.2),
                input_grad: (input_grad.1, input_grad.2),
                weight_grad: (weight_grad.1, weight_grad.2),
            },
            workspace,
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
        assert!(
            buffer.bytes() >= floats * 4,
            "the {what} buffer holds {} bytes, and the convolution needs {}",
            buffer.bytes(),
            floats * 4
        );
    }

    /// `Ok` for a cuDNN status of 0.
    fn check(&self, what: &'static str, status: c_int) -> Result<(), CudaError> {
        self.conv.check(what, status)
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
        self.cuda.bind()?;
        let api = &self.cuda.cudnn;
        // SAFETY: the handle and descriptors are live, every buffer holds its tensor
        // (checked above), the scratch buffer holds the algorithm's bytes, and the output,
        // borrowed mutably, is none of the inputs. With beta 0 the output is not read.
        let status = unsafe {
            (api.cudnnConvolutionForward)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.x.raw,
                x.pointer,
                self.w.raw,
                w.pointer,
                self.conv.raw,
                self.forward_algo,
                self.workspace.pointer,
                self.algorithms.forward.1,
                factor(&ZERO),
                self.y.raw,
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
                    self.bias.raw,
                    b.pointer,
                    factor(&ONE),
                    self.y.raw,
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
        self.cuda.bind()?;
        // SAFETY: as in `forward`, for the input-gradient algorithm.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardData)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.w.raw,
                w.pointer,
                self.y.raw,
                dy.pointer,
                self.conv.raw,
                self.input_grad_algo,
                self.workspace.pointer,
                self.algorithms.input_grad.1,
                factor(&ZERO),
                self.x.raw,
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
        self.cuda.bind()?;
        // SAFETY: as in `forward`, for the weight-gradient algorithm.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardFilter)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.x.raw,
                x.pointer,
                self.y.raw,
                dy.pointer,
                self.conv.raw,
                self.weight_grad_algo,
                self.workspace.pointer,
                self.algorithms.weight_grad.1,
                factor(&ZERO),
                self.w.raw,
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
        self.cuda.bind()?;
        // SAFETY: the handle and descriptors are live, both buffers hold their tensors
        // (checked above), and with beta 0 the gradient is not read.
        let status = unsafe {
            (self.cuda.cudnn.cudnnConvolutionBackwardBias)(
                self.cuda.cudnn_handle,
                factor(&ONE),
                self.y.raw,
                dy.pointer,
                factor(&ZERO),
                self.bias.raw,
                db.pointer,
            )
        };
        self.check("cudnnConvolutionBackwardBias", status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The first algorithm that is deterministic, without tensor cores and within the
    /// scratch limit is picked; the error names the ones passed over.
    #[test]
    fn picking_passes_over_what_does_not_qualify() {
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
        ];
        let big = |algo| Ok(if algo == 3 { MAX_WORKSPACE + 1 } else { 64 });
        assert_eq!(
            pick("weight-gradient", &WEIGHT_GRAD, &ranked, big).unwrap(),
            (5, "WINOGRAD_NONFUSED", 64)
        );
        let err = pick("weight-gradient", &WEIGHT_GRAD, &ranked[..4], big)
            .unwrap_err()
            .to_string();
        assert!(err.contains("it offered ALGO_0, FFT, ALGO_3"), "{err}");
    }
}
