//! 2-D max and average pooling, forward and backward, as this crate's own CUDA kernels, the
//! twins of the CPU's `max_pool2d_reference`, `max_pool2d_backward`, `avg_pool2d` and
//! `avg_pool2d_backward` in `cavitation_core::kernels::pooling`.

use std::fmt;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::module::{Arg, Module};
use crate::sys::CuFunction;

/// The kernels' source, a port of `cavitation-gpu`'s `kernels/pooling.rs`.
const SOURCE: &str = include_str!("pooling.cu");

/// Threads per block.
const BLOCK: u32 = 256;

/// The shape of a 2-D pooling over a batch of NCHW images, as the CPU's `Pool2dGeom` and a
/// batch size describe one; pooling keeps the channels.
///
/// The output size follows PyTorch, as the CPU computes it: along each axis,
/// `(size + 2 · padding − dilation · (kernel − 1) − 1) / stride + 1`, rounded down, or up
/// with `ceil_mode`, a last window that would start in the padding past the image dropped.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::PoolShape;
///
/// let shape = PoolShape::new(128, 32, (32, 32), (2, 2));
/// assert_eq!(shape.output(), Some((16, 16)));
/// assert_eq!(shape.y_len(), 128 * 32 * 16 * 16);
/// let ceil = PoolShape {
///     stride: (2, 2),
///     padding: (1, 1),
///     ceil_mode: true,
///     ..PoolShape::new(1, 1, (7, 7), (3, 3))
/// };
/// assert_eq!(ceil.output(), Some((4, 4)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolShape {
    /// Images in the batch.
    pub batch: u32,
    /// Channels of each image.
    pub channels: u32,
    /// Height and width of each input image.
    pub input: (u32, u32),
    /// Window height and width.
    pub kernel: (u32, u32),
    /// Step between windows, down and across.
    pub stride: (u32, u32),
    /// Positions added on each side, top and bottom, left and right: zeros for average
    /// pooling, positions that never win for max pooling.
    pub padding: (u32, u32),
    /// Spacing between the window's taps, down and across (max pooling only).
    pub dilation: (u32, u32),
    /// Whether the output size rounds up.
    pub ceil_mode: bool,
}

impl PoolShape {
    /// A pooling with the stride equal to the window, no padding, no dilation and
    /// `ceil_mode` off: PyTorch's defaults.
    pub fn new(batch: u32, channels: u32, input: (u32, u32), kernel: (u32, u32)) -> PoolShape {
        PoolShape {
            batch,
            channels,
            input,
            kernel,
            stride: kernel,
            padding: (0, 0),
            dilation: (1, 1),
            ceil_mode: false,
        }
    }

    /// The axes as `(size, kernel, stride, padding, dilation)`, height first.
    fn axes(&self) -> [(u32, u32, u32, u32, u32); 2] {
        [
            (
                self.input.0,
                self.kernel.0,
                self.stride.0,
                self.padding.0,
                self.dilation.0,
            ),
            (
                self.input.1,
                self.kernel.1,
                self.stride.1,
                self.padding.1,
                self.dilation.1,
            ),
        ]
    }

    /// The output's height and width, or `None` when no window fits or a window, stride or
    /// dilation is zero.
    pub fn output(&self) -> Option<(u32, u32)> {
        let ceil = self.ceil_mode;
        let along = |(size, kernel, stride, padding, dilation): (u32, u32, u32, u32, u32)| {
            if kernel == 0 || stride == 0 || dilation == 0 {
                return None;
            }
            let (size, stride, padding) = (u64::from(size), u64::from(stride), u64::from(padding));
            let span = u64::from(dilation) * u64::from(kernel - 1) + 1;
            let room = (size + 2 * padding).checked_sub(span)?;
            let mut out = if ceil {
                room.div_ceil(stride) + 1
            } else {
                room / stride + 1
            };
            if ceil && (out - 1) * stride >= size + padding {
                out -= 1;
            }
            u32::try_from(out).ok()
        };
        let [rows, cols] = self.axes();
        Some((along(rows)?, along(cols)?))
    }

    /// The product of four sizes, or `None` if it does not fit a `usize`.
    fn product(sizes: [u32; 4]) -> Option<usize> {
        sizes
            .iter()
            .try_fold(1usize, |acc, &s| acc.checked_mul(s as usize))
    }

    /// The input's length in floats: batch × channels × height × width, saturating at
    /// `usize::MAX` (the kernels refuse a shape whose lengths do not fit a `u32`).
    pub fn x_len(&self) -> usize {
        Self::product([self.batch, self.channels, self.input.0, self.input.1]).unwrap_or(usize::MAX)
    }

    /// The output's length in floats (and the winners' length in `u32`s), 0 when no window
    /// fits, saturating at `usize::MAX`.
    pub fn y_len(&self) -> usize {
        self.output().map_or(0, |(h, w)| {
            Self::product([self.batch, self.channels, h, w]).unwrap_or(usize::MAX)
        })
    }

    /// The output size, after checking the shape as the CPU's `Pool2dGeom` checks it and
    /// that every index the kernels form fits a `u32`.
    fn check(&self, what: &str) -> Result<(u32, u32), CudaError> {
        let refuse = |why: String| Err(CudaError::Invalid(format!("{what}: {why}: {self:?}")));
        if [self.batch, self.channels, self.input.0, self.input.1].contains(&0) {
            return refuse("the batch, channels and image must be at least 1".into());
        }
        if self.padding.0 > self.kernel.0 / 2 || self.padding.1 > self.kernel.1 / 2 {
            return refuse("the padding must be at most half the window".into());
        }
        let Some((oh, ow)) = self.output() else {
            return refuse("no window fits, or a window, stride or dilation is 0".into());
        };
        // With dilation a window's taps can all miss the image; the CPU refuses that.
        let misses =
            |out: u32, (size, kernel, stride, padding, dilation): (u32, u32, u32, u32, u32)| {
                (0..u64::from(out)).any(|o| {
                    (0..u64::from(kernel)).all(|k| {
                        (o * u64::from(stride) + k * u64::from(dilation))
                            .checked_sub(u64::from(padding))
                            .is_none_or(|v| v >= u64::from(size))
                    })
                })
            };
        let [rows, cols] = self.axes();
        if misses(oh, rows) || misses(ow, cols) {
            return refuse("a window holds no pixel of the image".into());
        }
        // Every padded position and every intermediate of the window arithmetic is below
        // the padded size plus a dilated window and a stride.
        let reach = |(size, kernel, stride, padding, dilation): (u32, u32, u32, u32, u32)| {
            u64::from(size)
                + 2 * u64::from(padding)
                + u64::from(dilation) * u64::from(kernel)
                + u64::from(stride)
        };
        let limit = u64::from(u32::MAX);
        if reach(rows) > limit || reach(cols) > limit {
            return refuse("the window arithmetic does not fit a u32".into());
        }
        if self.x_len() > limit as usize || self.y_len() > limit as usize {
            return refuse("the input or output holds more than 2³² − 1 values".into());
        }
        Ok((oh, ow))
    }

    /// The macros the kernels read the geometry from, for the output size `(oh, ow)`: the
    /// threads of each pass, the sizes, the window, the stride and the padding.
    fn defines(&self, (oh, ow): (u32, u32)) -> String {
        let mut text = String::new();
        for (name, value) in [
            ("FORWARD_TOTAL", self.y_len() as u64),
            ("BACKWARD_TOTAL", self.x_len() as u64),
            ("H", u64::from(self.input.0)),
            ("W", u64::from(self.input.1)),
            ("OH", u64::from(oh)),
            ("OW", u64::from(ow)),
            ("KH", u64::from(self.kernel.0)),
            ("KW", u64::from(self.kernel.1)),
            ("SH", u64::from(self.stride.0)),
            ("SW", u64::from(self.stride.1)),
            ("PH", u64::from(self.padding.0)),
            ("PW", u64::from(self.padding.1)),
        ] {
            text.push_str(&format!("#define {name} {value}u\n"));
        }
        text
    }
}

/// Panics unless every buffer holds at least `values` four-byte values, naming the first
/// that does not.
fn check_buffers(what: &str, buffers: &[(&DeviceBuffer<'_>, &str, usize)]) {
    for &(buffer, name, values) in buffers {
        let needs = values * 4;
        assert!(
            buffer.bytes() >= needs,
            "{what}: the {name} buffer holds {} bytes, and {values} values need {needs}",
            buffer.bytes()
        );
    }
}

/// A pooling's kernels compiled for one geometry: the module, the forward and backward
/// kernels' handles and the thread counts.
struct Compiled<'c> {
    /// The kernels, compiled for one geometry.
    module: Module<'c>,
    /// The forward kernel's handle.
    forward: CuFunction,
    /// The backward kernel's handle.
    backward: CuFunction,
    /// Outputs in the batch: one forward thread each.
    outputs: usize,
    /// Input pixels in the batch: one backward thread each.
    inputs: usize,
}

impl<'c> Compiled<'c> {
    /// Compiles the source as `name` with `defines` in front, for `shape` (checked); the
    /// forward kernel is `prefix` + `_forward`, the backward one `prefix` + `_backward`.
    fn new(
        cuda: &'c Cuda,
        name: &str,
        prefix: &str,
        defines: String,
        shape: &PoolShape,
    ) -> Result<Compiled<'c>, CudaError> {
        let module = cuda.compile(name, &(defines + SOURCE))?;
        let forward = module.function(&format!("{prefix}_forward"))?;
        let backward = module.function(&format!("{prefix}_backward"))?;
        Ok(Compiled {
            module,
            forward,
            backward,
            outputs: shape.y_len(),
            inputs: shape.x_len(),
        })
    }

    /// Launches `function`, one of this module's handles, on `threads` threads with `args`.
    ///
    /// # Safety
    ///
    /// `args` must be the kernel's parameters, and every buffer must hold what the kernel
    /// reads and writes for `threads` threads.
    unsafe fn launch(
        &self,
        function: CuFunction,
        threads: usize,
        args: &[Arg<'_>],
    ) -> Result<(), CudaError> {
        // SAFETY: both handles were looked up in this module when it was compiled.
        let kernel = unsafe { self.module.kernel_from(function) };
        // The thread count fits a u32 (checked when the shape was), so the blocks do too.
        let blocks = (threads as u32).div_ceil(BLOCK);
        // SAFETY: the caller passes the kernel's parameters and buffers that hold what it
        // touches for `threads` threads, the kernel's total.
        unsafe { kernel.launch([blocks, 1, 1], [BLOCK, 1, 1], args) }
    }
}

/// Max pooling on the card for one shape, as the CPU's `max_pool2d_reference` computes it,
/// bit for bit: each output is its window's largest value, the taps read in row-major order,
/// a later tap winning when it is greater or NaN (so the last NaN of a window wins), padding
/// never winning, and the winner's bits copied unchanged (signed zeros, subnormals and NaN
/// payloads included). Each winner is the winner's position in its plane
/// (`row · width + column`), as a `u32`.
///
/// The backward pass adds each output's gradient to its winner's input gradient, one thread
/// per input pixel adding its outputs in the CPU's order, so it gives `max_pool2d_backward`'s
/// bits, subnormal sums included, and the same bits on every run.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{Cuda, MaxPool2d, PoolShape};
///
/// if let Some(cuda) = Cuda::new() {
///     let pool = MaxPool2d::new(&cuda, PoolShape::new(1, 1, (2, 2), (2, 2))).unwrap();
///     let mut x = cuda.buffer(16).unwrap();
///     let (mut y, mut winners) = (cuda.buffer(4).unwrap(), cuda.buffer(4).unwrap());
///     x.write(&[1.0f32, 4.0, 3.0, 2.0]).unwrap();
///     pool.forward(&x, &mut y, &mut winners).unwrap();
///     let (mut out, mut at) = ([0.0f32], [0u32]);
///     y.read(&mut out).unwrap();
///     winners.read(&mut at).unwrap();
///     assert_eq!((out, at), ([4.0], [1]));
/// }
/// ```
pub struct MaxPool2d<'c> {
    /// The shape.
    shape: PoolShape,
    /// The kernels, compiled for the shape.
    kernels: Compiled<'c>,
}

impl fmt::Debug for MaxPool2d<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaxPool2d")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl<'c> MaxPool2d<'c> {
    /// Compiles the forward and backward kernels for `shape` on `cuda`'s card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] for a shape the CPU's `Pool2dGeom` refuses (a size of 0, the
    /// padding more than half the window, no window fitting, a window that holds no pixel of
    /// the image) or whose input, output or window arithmetic does not fit a `u32`;
    /// [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the driver refuses the
    /// kernels.
    pub fn new(cuda: &'c Cuda, shape: PoolShape) -> Result<MaxPool2d<'c>, CudaError> {
        let output = shape.check("max_pool2d")?;
        let defines = format!(
            "{}#define DH {}u\n#define DW {}u\n",
            shape.defines(output),
            shape.dilation.0,
            shape.dilation.1
        );
        let kernels = Compiled::new(cuda, "pooling.cu (max)", "max", defines, &shape)?;
        Ok(MaxPool2d { shape, kernels })
    }

    /// The shape.
    pub fn shape(&self) -> PoolShape {
        self.shape
    }

    /// Writes each output of the batch `x` to `y` (`f32`) and its winner to `winners`
    /// (`u32`), every one of them.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `x` holds fewer than [`PoolShape::x_len`] floats or `y` or `winners` fewer than
    /// [`PoolShape::y_len`] values.
    pub fn forward(
        &self,
        x: &DeviceBuffer<'_>,
        y: &mut DeviceBuffer<'_>,
        winners: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let k = &self.kernels;
        check_buffers(
            "max_pool2d forward",
            &[
                (x, "input", k.inputs),
                (y, "output", k.outputs),
                (winners, "winners", k.outputs),
            ],
        );
        // SAFETY: the forward kernel takes (const u32 *x, u32 *out, u32 *winners), reads
        // inputs below `inputs` and writes outputs below `outputs`, which the buffers hold
        // (checked above); y and winners, borrowed mutably, are distinct from each other and
        // from x.
        unsafe {
            k.launch(
                k.forward,
                k.outputs,
                &[Arg::Buffer(x), Arg::Buffer(y), Arg::Buffer(winners)],
            )
        }
    }

    /// Adds each output gradient `dy` to the input gradient `dx` at its output's winner, from
    /// the `winners` the forward pass wrote.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `dy` or `winners` holds fewer than [`PoolShape::y_len`] values or `dx` fewer than
    /// [`PoolShape::x_len`] floats.
    pub fn backward(
        &self,
        dy: &DeviceBuffer<'_>,
        winners: &DeviceBuffer<'_>,
        dx: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let k = &self.kernels;
        check_buffers(
            "max_pool2d backward",
            &[
                (dy, "output gradient", k.outputs),
                (winners, "winners", k.outputs),
                (dx, "input gradient", k.inputs),
            ],
        );
        // SAFETY: the backward kernel takes (const float *grad_out, const u32 *winners, float
        // *grad_in), reads outputs below `outputs` and touches input pixels below `inputs`,
        // which the buffers hold (checked above); dx, borrowed mutably, is neither input.
        unsafe {
            k.launch(
                k.backward,
                k.inputs,
                &[Arg::Buffer(dy), Arg::Buffer(winners), Arg::Buffer(dx)],
            )
        }
    }
}

/// Average pooling on the card for one shape and divisor rule, as the CPU's `avg_pool2d` and
/// `avg_pool2d_backward` compute it, bit for bit: each output is its window's in-image sum,
/// in row-major order, divided by `divisor_override` if given, else by the window's size
/// counting the padding (`count_include_pad`), else by the pixels summed; the backward pass
/// adds each output gradient divided by its divisor to every in-image pixel of its window,
/// one thread per pixel adding its outputs in the CPU's order. The division is correctly
/// rounded and nothing is flushed, so subnormals, ±∞ and NaN go where the CPU's do.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::{AvgPool2d, Cuda, PoolShape};
///
/// if let Some(cuda) = Cuda::new() {
///     let pool = AvgPool2d::new(&cuda, PoolShape::new(1, 1, (2, 2), (2, 2)), true, None).unwrap();
///     let (mut x, mut y) = (cuda.buffer(16).unwrap(), cuda.buffer(4).unwrap());
///     x.write(&[1.0f32, 4.0, 3.0, 2.0]).unwrap();
///     pool.forward(&x, &mut y).unwrap();
///     let mut out = [0.0f32];
///     y.read(&mut out).unwrap();
///     assert_eq!(out, [2.5]);
/// }
/// ```
pub struct AvgPool2d<'c> {
    /// The shape.
    shape: PoolShape,
    /// The kernels, compiled for the shape and divisor rule.
    kernels: Compiled<'c>,
}

impl fmt::Debug for AvgPool2d<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AvgPool2d")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl<'c> AvgPool2d<'c> {
    /// Compiles the forward and backward kernels for `shape` on `cuda`'s card, dividing by
    /// `divisor_override` if given, else by the window counting the padding when
    /// `count_include_pad`, else by the pixels summed.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] as for [`MaxPool2d::new`], and for a dilated shape, a divisor
    /// override of 0 or a window whose size does not fit a `u32`, as the CPU's `avg_pool2d`
    /// refuses the first two; [`CudaError::Compile`] or [`CudaError::Call`] if NVRTC or the
    /// driver refuses the kernels.
    pub fn new(
        cuda: &'c Cuda,
        shape: PoolShape,
        count_include_pad: bool,
        divisor_override: Option<u32>,
    ) -> Result<AvgPool2d<'c>, CudaError> {
        if shape.dilation != (1, 1) {
            return Err(CudaError::Invalid(format!(
                "avg_pool2d: average pooling has no dilation, got {:?}",
                shape.dilation
            )));
        }
        if divisor_override == Some(0) {
            return Err(CudaError::Invalid(
                "avg_pool2d: the divisor override must not be 0".into(),
            ));
        }
        if shape.kernel.0.checked_mul(shape.kernel.1).is_none() {
            return Err(CudaError::Invalid(format!(
                "avg_pool2d: the window {:?} holds more than 2³² − 1 values",
                shape.kernel
            )));
        }
        let output = shape.check("avg_pool2d")?;
        let defines = format!(
            "{}#define COUNT_PAD {}\n#define DIVISOR {}u\n",
            shape.defines(output),
            u8::from(count_include_pad),
            divisor_override.unwrap_or(0)
        );
        let kernels = Compiled::new(cuda, "pooling.cu (avg)", "avg", defines, &shape)?;
        Ok(AvgPool2d { shape, kernels })
    }

    /// The shape.
    pub fn shape(&self) -> PoolShape {
        self.shape
    }

    /// Writes each output of the batch `x` to `y`, every one of them.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `x` holds fewer than [`PoolShape::x_len`] floats or `y` fewer than
    /// [`PoolShape::y_len`].
    pub fn forward(&self, x: &DeviceBuffer<'_>, y: &mut DeviceBuffer<'_>) -> Result<(), CudaError> {
        let k = &self.kernels;
        check_buffers(
            "avg_pool2d forward",
            &[(x, "input", k.inputs), (y, "output", k.outputs)],
        );
        // SAFETY: the forward kernel takes (const float *x, float *out), reads inputs below
        // `inputs` and writes outputs below `outputs`, which the buffers hold (checked
        // above); y, borrowed mutably, is not x.
        unsafe { k.launch(k.forward, k.outputs, &[Arg::Buffer(x), Arg::Buffer(y)]) }
    }

    /// Adds each output gradient `dy`, divided by its divisor, to the input gradient `dx` at
    /// every in-image pixel of its window.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the launch is refused.
    ///
    /// # Panics
    ///
    /// If `dy` holds fewer than [`PoolShape::y_len`] floats or `dx` fewer than
    /// [`PoolShape::x_len`].
    pub fn backward(
        &self,
        dy: &DeviceBuffer<'_>,
        dx: &mut DeviceBuffer<'_>,
    ) -> Result<(), CudaError> {
        let k = &self.kernels;
        check_buffers(
            "avg_pool2d backward",
            &[
                (dy, "output gradient", k.outputs),
                (dx, "input gradient", k.inputs),
            ],
        );
        // SAFETY: the backward kernel takes (const float *grad_out, float *grad_in), reads
        // outputs below `outputs` and touches input pixels below `inputs`, which the buffers
        // hold (checked above); dx, borrowed mutably, is not dy.
        unsafe { k.launch(k.backward, k.inputs, &[Arg::Buffer(dy), Arg::Buffer(dx)]) }
    }
}
