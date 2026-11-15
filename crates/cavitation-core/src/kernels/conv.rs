//! 2-D convolution: its geometry, the column layout the fast path multiplies, and the direct
//! loops it is tested against.
//!
//! Images are flat rows in PyTorch's NCHW order: image `n` holds its channels one after
//! another, each `height` rows of `width` values. A weight is `[out, in, kh, kw]`, also as
//! PyTorch stores it.
//!
//! The fast path ([`Tape::conv2d`](crate::Tape::conv2d)) copies every window of the batch
//! into a column matrix ([`im2col`]) and multiplies it by the weight with the packed GEMM;
//! [`col2im`] adds a column matrix's gradient back onto the image. The direct loops
//! ([`conv2d_reference`], [`conv2d_backward_reference`]) are its reference twin.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_reference};
//! // One 3 x 3 image, one 2 x 2 kernel of ones: each output is the sum of a window.
//! let geom = Conv2dGeom::new(1, 1, (3, 3), (2, 2)).unwrap();
//! let x: Vec<f64> = (1..=9).map(f64::from).collect();
//! let mut out = vec![0.0; 4];
//! conv2d_reference(&geom, 1, &x, &[1.0; 4], None, &mut out);
//! assert_eq!(out, [12.0, 16.0, 24.0, 28.0]);
//! ```

use crate::kernels::{Level, MatMut, MatRef, direct};
use crate::{Error, Real};

/// The shape of a 2-D convolution: input channels and image size, output channels, kernel,
/// stride, zero padding and dilation (each per axis, height first), checked when it is made
/// so that every image gives a non-empty output.
///
/// The output size follows PyTorch's `nn.Conv2d`: along each axis,
/// `(size + 2 · padding − dilation · (kernel − 1) − 1) / stride + 1`, rounded down.
///
/// ```
/// use cavitation_core::kernels::conv::Conv2dGeom;
/// let geom = Conv2dGeom::new(3, 16, (32, 32), (3, 3))
///     .unwrap()
///     .with_padding((1, 1))
///     .unwrap();
/// assert_eq!(geom.out_size(), (32, 32));
/// assert_eq!((geom.in_width(), geom.out_width()), (3 * 32 * 32, 16 * 32 * 32));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Conv2dGeom {
    /// Channels of each input image.
    in_channels: usize,
    /// Channels of each output image: the number of kernels.
    out_channels: usize,
    /// Height and width of each input image.
    input: (usize, usize),
    /// Kernel height and width.
    kernel: (usize, usize),
    /// Step between windows, down and across.
    stride: (usize, usize),
    /// Zeros added on each side, top and bottom, left and right.
    padding: (usize, usize),
    /// Spacing between the kernel's taps, down and across.
    dilation: (usize, usize),
}

/// The output size along one axis, or `None` if no window fits.
fn axis_out(
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let span = dilation.checked_mul(kernel - 1)?.checked_add(1)?;
    let padded = size.checked_add(padding.checked_mul(2)?)?;
    padded.checked_sub(span).map(|room| room / stride + 1)
}

impl Conv2dGeom {
    /// A convolution of `in_channels` images of `input` (height, width) by `out_channels`
    /// kernels of `kernel` (height, width), with stride 1, no padding and no dilation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if any count or size is 0, or the kernel is larger than the
    /// image, naming the sizes.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(1, 6, (28, 28), (5, 5)).unwrap().out_size(), (24, 24));
    /// assert!(Conv2dGeom::new(1, 6, (4, 4), (5, 5)).is_err());
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        input: (usize, usize),
        kernel: (usize, usize),
    ) -> Result<Conv2dGeom, Error> {
        Conv2dGeom {
            in_channels,
            out_channels,
            input,
            kernel,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
        .checked()
    }

    /// A convolution with every setting given at once: stride, zero padding and dilation
    /// (each down, then across). Use this when the kernel fits the image only with its
    /// padding, which [`new`](Self::new) followed by [`with_padding`](Self::with_padding)
    /// cannot express.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] as [`new`](Self::new) and the `with_` methods describe.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// // A 1 x 1 image fits a 3 x 3 kernel once it is padded by 1.
    /// let geom = Conv2dGeom::with_options(4, 8, (1, 1), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
    /// assert_eq!(geom.out_size(), (1, 1));
    /// ```
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        input: (usize, usize),
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Conv2dGeom, Error> {
        Conv2dGeom {
            in_channels,
            out_channels,
            input,
            kernel,
            stride,
            padding,
            dilation,
        }
        .checked()
    }

    /// This geometry with stride `stride` (down, across).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if either step is 0.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// let geom = Conv2dGeom::new(1, 1, (7, 7), (3, 3)).unwrap().with_stride((2, 2)).unwrap();
    /// assert_eq!(geom.out_size(), (3, 3));
    /// ```
    pub fn with_stride(self, stride: (usize, usize)) -> Result<Conv2dGeom, Error> {
        Conv2dGeom { stride, ..self }.checked()
    }

    /// This geometry with `padding` zeros on each side (top and bottom, left and right).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if the padded size overflows.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// let geom = Conv2dGeom::new(1, 1, (5, 5), (3, 3)).unwrap().with_padding((1, 1)).unwrap();
    /// assert_eq!(geom.out_size(), (5, 5));
    /// ```
    pub fn with_padding(self, padding: (usize, usize)) -> Result<Conv2dGeom, Error> {
        Conv2dGeom { padding, ..self }.checked()
    }

    /// This geometry with dilation `dilation` (down, across): a kernel tap every
    /// `dilation` pixels.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if either spacing is 0 or the dilated kernel no longer fits.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// let geom = Conv2dGeom::new(1, 1, (7, 7), (3, 3)).unwrap().with_dilation((2, 2)).unwrap();
    /// assert_eq!(geom.out_size(), (3, 3));
    /// ```
    pub fn with_dilation(self, dilation: (usize, usize)) -> Result<Conv2dGeom, Error> {
        Conv2dGeom { dilation, ..self }.checked()
    }

    /// `self` if it is valid, else the error naming what is wrong.
    fn checked(self) -> Result<Conv2dGeom, Error> {
        let fail = |what: String| Err(Error::InvalidData(format!("conv2d: {what}")));
        let Conv2dGeom {
            in_channels,
            out_channels,
            input,
            kernel,
            stride,
            dilation,
            padding,
        } = self;
        if in_channels == 0 || out_channels == 0 {
            return fail(format!(
                "the channel counts must be at least 1, got {in_channels} in and {out_channels} out"
            ));
        }
        if input.0 == 0 || input.1 == 0 || kernel.0 == 0 || kernel.1 == 0 {
            return fail(format!(
                "the image and kernel sizes must be at least 1, got image {input:?} and \
                 kernel {kernel:?}"
            ));
        }
        if stride.0 == 0 || stride.1 == 0 || dilation.0 == 0 || dilation.1 == 0 {
            return fail(format!(
                "the stride and dilation must be at least 1, got stride {stride:?} and \
                 dilation {dilation:?}"
            ));
        }
        let rows = axis_out(input.0, kernel.0, stride.0, padding.0, dilation.0);
        let cols = axis_out(input.1, kernel.1, stride.1, padding.1, dilation.1);
        let (Some(_), Some(_)) = (rows, cols) else {
            return fail(format!(
                "a {kernel:?} kernel with dilation {dilation:?} does not fit an image of \
                 {input:?} with padding {padding:?}"
            ));
        };
        // Every size the operations multiply must fit in `usize`.
        let sizes = [
            in_channels
                .checked_mul(input.0)
                .and_then(|v| v.checked_mul(input.1)),
            in_channels
                .checked_mul(kernel.0)
                .and_then(|v| v.checked_mul(kernel.1)),
            self.out_channels
                .checked_mul(rows.unwrap_or(0))
                .and_then(|v| v.checked_mul(cols.unwrap_or(0))),
            // The weight, and one image's column matrix.
            in_channels
                .checked_mul(kernel.0)
                .and_then(|v| v.checked_mul(kernel.1))
                .and_then(|v| v.checked_mul(out_channels)),
            in_channels
                .checked_mul(kernel.0)
                .and_then(|v| v.checked_mul(kernel.1))
                .and_then(|v| v.checked_mul(rows.unwrap_or(0)))
                .and_then(|v| v.checked_mul(cols.unwrap_or(0))),
        ];
        if sizes.iter().any(Option::is_none) {
            return fail(format!("the sizes overflow: {self:?}"));
        }
        Ok(self)
    }

    /// Channels of each input image.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 4), (3, 3)).unwrap().in_channels(), 3);
    /// ```
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Channels of each output image.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 4), (3, 3)).unwrap().out_channels(), 8);
    /// ```
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Height and width of each input image.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 3)).unwrap().input(), (4, 5));
    /// ```
    pub fn input(&self) -> (usize, usize) {
        self.input
    }

    /// Kernel height and width.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 2)).unwrap().kernel(), (3, 2));
    /// ```
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    /// The stride, down and across.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(1, 1, (4, 4), (3, 3)).unwrap().stride(), (1, 1));
    /// ```
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// The zero padding on each side, top and bottom, left and right.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(1, 1, (4, 4), (3, 3)).unwrap().padding(), (0, 0));
    /// ```
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// The dilation, down and across.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(1, 1, (4, 4), (3, 3)).unwrap().dilation(), (1, 1));
    /// ```
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// Height and width of each output image.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(1, 1, (6, 5), (3, 2)).unwrap().out_size(), (4, 4));
    /// ```
    pub fn out_size(&self) -> (usize, usize) {
        let along = |a: usize| -> usize {
            let (size, kernel, stride, padding, dilation) = if a == 0 {
                (
                    self.input.0,
                    self.kernel.0,
                    self.stride.0,
                    self.padding.0,
                    self.dilation.0,
                )
            } else {
                (
                    self.input.1,
                    self.kernel.1,
                    self.stride.1,
                    self.padding.1,
                    self.dilation.1,
                )
            };
            axis_out(size, kernel, stride, padding, dilation).expect("checked when made")
        };
        (along(0), along(1))
    }

    /// Values in one input image: `in_channels · height · width`, the width of an input row.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 3)).unwrap().in_width(), 60);
    /// ```
    pub fn in_width(&self) -> usize {
        self.in_channels * self.input.0 * self.input.1
    }

    /// Values in one output image: `out_channels · out_height · out_width`.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 3)).unwrap().out_width(), 8 * 2 * 3);
    /// ```
    pub fn out_width(&self) -> usize {
        let (oh, ow) = self.out_size();
        self.out_channels * oh * ow
    }

    /// Values in one kernel: `in_channels · kh · kw`, the length of each output's sum and
    /// the number of rows of the column matrix.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 2)).unwrap().kernel_len(), 18);
    /// ```
    pub fn kernel_len(&self) -> usize {
        self.in_channels * self.kernel.0 * self.kernel.1
    }

    /// Values in the weight: `out_channels · kernel_len`.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert_eq!(Conv2dGeom::new(3, 8, (4, 5), (3, 2)).unwrap().weight_len(), 144);
    /// ```
    pub fn weight_len(&self) -> usize {
        self.out_channels * self.kernel_len()
    }

    /// Whether the convolution is a plain matrix product per image: a 1 × 1 kernel with
    /// stride 1 and no padding, whose column matrix is the image itself.
    ///
    /// ```
    /// use cavitation_core::kernels::conv::Conv2dGeom;
    /// assert!(Conv2dGeom::new(8, 4, (5, 5), (1, 1)).unwrap().is_pointwise());
    /// assert!(!Conv2dGeom::new(8, 4, (5, 5), (3, 3)).unwrap().is_pointwise());
    /// ```
    pub fn is_pointwise(&self) -> bool {
        self.kernel == (1, 1) && self.stride == (1, 1) && self.padding == (0, 0)
    }

    /// The input position read by output position `o` (along one axis) at kernel tap `t`,
    /// or `None` where it falls in the padding.
    #[inline]
    fn source(
        o: usize,
        t: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        size: usize,
    ) -> Option<usize> {
        (o * stride + t * dilation)
            .checked_sub(padding)
            .filter(|&s| s < size)
    }

    /// The geometry as twelve numbers, in the order [`from_words`](Self::from_words) reads.
    pub(crate) fn to_words(self) -> [usize; 12] {
        [
            self.in_channels,
            self.out_channels,
            self.input.0,
            self.input.1,
            self.kernel.0,
            self.kernel.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.dilation.0,
            self.dilation.1,
        ]
    }

    /// The geometry [`to_words`](Self::to_words) wrote.
    pub(crate) fn from_words(w: &[usize]) -> Conv2dGeom {
        Conv2dGeom {
            in_channels: w[0],
            out_channels: w[1],
            input: (w[2], w[3]),
            kernel: (w[4], w[5]),
            stride: (w[6], w[7]),
            padding: (w[8], w[9]),
            dilation: (w[10], w[11]),
        }
    }
}

/// Panics, naming `what`, unless `len` is `expected`.
fn check_len(what: &str, len: usize, expected: usize) {
    assert!(
        len == expected,
        "conv2d: {what} holds {len} values, expected {expected}"
    );
}

/// Copies every window of `batch` images `x` into the column matrix `cols`, row-major
/// `[kernel_len, batch · out_h · out_w]`: row `(c · kh + i) · kw + j` holds tap `(c, i, j)`
/// of every window, image after image, each image's windows in row-major output order, with
/// 0 where the tap falls in the padding. The product `weight [out, kernel_len] · cols` is
/// then every output of the batch.
///
/// # Panics
///
/// Panics if `x` or `cols` has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, im2col};
/// let geom = Conv2dGeom::new(1, 1, (2, 2), (1, 2)).unwrap();
/// let mut cols = vec![0.0f32; 2 * 2];
/// im2col(&geom, 1, &[1.0, 2.0, 3.0, 4.0], &mut cols);
/// // Two windows (one per row); the first tap row holds their left pixels.
/// assert_eq!(cols, [1.0, 3.0, 2.0, 4.0]);
/// ```
pub fn im2col<T: Real>(geom: &Conv2dGeom, batch: usize, x: &[T], cols: &mut [T]) {
    check_len("the input", x.len(), batch * geom.in_width());
    check_len(
        "the column matrix",
        cols.len(),
        geom.kernel_len() * batch * geom.out_size().0 * geom.out_size().1,
    );
    unroll(geom, batch, x, (0, geom.out_size().0), cols);
}

/// [`im2col`] for output rows `r0..r0 + nr` of each of the `batch` images `x` only: `cols`
/// is `[kernel_len, batch · nr · out_w]`, each image's `nr` rows of windows in row-major
/// order. Lengths are the caller's to check.
fn unroll<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    x: &[T],
    (r0, nr): (usize, usize),
    cols: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let windows = nr * ow;
    let n_cols = batch * windows;
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dw)) = (geom.stride, geom.padding, geom.dilation);
    for c in 0..geom.in_channels {
        for i in 0..kh {
            let rows = span(oh, i, sh, ph, dh, h);
            for j in 0..kw {
                let across = span(ow, j, sw, pw, dw, w);
                let row = &mut cols[((c * kh + i) * kw + j) * n_cols..][..n_cols];
                for n in 0..batch {
                    let plane = &x[(n * geom.in_channels + c) * h * w..][..h * w];
                    let dst = &mut row[n * windows..][..windows];
                    for oy in r0..r0 + nr {
                        let line = &mut dst[(oy - r0) * ow..][..ow];
                        if !rows.contains(&oy) {
                            line.fill(T::ZERO);
                            continue;
                        }
                        let y = oy * sh + i * dh - ph;
                        let src = &plane[y * w..][..w];
                        line[..across.start].fill(T::ZERO);
                        line[across.end..].fill(T::ZERO);
                        if across.is_empty() {
                            continue;
                        }
                        let first = across.start * sw + j * dw - pw;
                        if sw == 1 {
                            line[across.clone()].copy_from_slice(&src[first..][..across.len()]);
                        } else {
                            for (k, v) in line[across.clone()].iter_mut().enumerate() {
                                *v = src[first + k * sw];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The output positions `lo..hi` along one axis whose tap `t` falls inside the image (not
/// in the padding): those `o` with `0 <= o · stride + t · dilation − padding < size`.
fn span(
    out: usize,
    t: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size: usize,
) -> std::ops::Range<usize> {
    let offset = t * dilation;
    // The first `o` with o · stride + offset >= padding.
    let lo = if offset >= padding {
        0
    } else {
        (padding - offset).div_ceil(stride)
    };
    // One past the last `o` with o · stride + offset < padding + size.
    let limit = padding + size;
    let hi = if offset >= limit {
        0
    } else {
        ((limit - offset - 1) / stride + 1).min(out)
    };
    lo.min(hi)..hi
}

/// Adds the column matrix `cols` (laid out as [`im2col`] writes it) back onto the `batch`
/// images `dx`: each tap's value is added to the input position it was read from, taps in
/// the padding dropped. This is the gradient of `im2col`. Each input position receives its
/// contributions in a fixed order (by channel, then kernel row, then kernel column, then
/// output position), so the result does not depend on anything but the inputs.
///
/// # Panics
///
/// Panics if `dx` or `cols` has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, col2im};
/// // Two overlapping 1 x 2 windows on a 1 x 3 row: the middle pixel is read twice.
/// let geom = Conv2dGeom::new(1, 1, (1, 3), (1, 2)).unwrap();
/// let mut dx = vec![0.0f32; 3];
/// col2im(&geom, 1, &[1.0, 1.0, 1.0, 1.0], &mut dx);
/// assert_eq!(dx, [1.0, 2.0, 1.0]);
/// ```
pub fn col2im<T: Real>(geom: &Conv2dGeom, batch: usize, cols: &[T], dx: &mut [T]) {
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    check_len(
        "the column matrix",
        cols.len(),
        geom.kernel_len() * batch * geom.out_size().0 * geom.out_size().1,
    );
    roll_back(geom, batch, cols, (0, geom.out_size().0), dx);
}

/// [`col2im`] for a column matrix of output rows `r0..r0 + nr` of each image only, laid out
/// as [`unroll`] writes it: adds each tap's value to the input position it was read from,
/// by channel, then kernel row, then kernel column, then output position. Lengths are the
/// caller's to check.
fn roll_back<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    cols: &[T],
    (r0, nr): (usize, usize),
    dx: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let windows = nr * ow;
    let n_cols = batch * windows;
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dw)) = (geom.stride, geom.padding, geom.dilation);
    for n in 0..batch {
        for c in 0..geom.in_channels {
            let plane = &mut dx[(n * geom.in_channels + c) * h * w..][..h * w];
            for i in 0..kh {
                let all = span(oh, i, sh, ph, dh, h);
                let rows = all.start.max(r0)..all.end.min(r0 + nr);
                for j in 0..kw {
                    let across = span(ow, j, sw, pw, dw, w);
                    if across.is_empty() {
                        continue;
                    }
                    let src = &cols[((c * kh + i) * kw + j) * n_cols + n * windows..][..windows];
                    let first = across.start * sw + j * dw - pw;
                    for oy in rows.clone() {
                        let y = oy * sh + i * dh - ph;
                        let line = &src[(oy - r0) * ow + across.start..][..across.len()];
                        let dst = &mut plane[y * w..][..w];
                        for (k, &v) in line.iter().enumerate() {
                            dst[first + k * sw] += v;
                        }
                    }
                }
            }
        }
    }
}

/// The most groups a batch is cut into: groups of consecutive images run as separate tasks
/// on the thread pool, so a batch of `n` images is `min(n, CONV_GROUPS)` groups whatever the
/// thread count, and every result (the weight gradient is a sum of one partial per group,
/// added in group order) is the same for any thread count.
///
/// ```
/// use cavitation_core::kernels::conv::{CONV_GROUPS, groups};
/// assert_eq!(groups(1000), CONV_GROUPS);
/// ```
pub const CONV_GROUPS: usize = 16;

/// How many column-matrix values one group aims to hold at once (512 KB in f32): a group
/// takes its images in chunks of `max(1, COLUMN_BUDGET / (kernel_len · windows))`, so each
/// GEMM is wide enough for the packed kernels while every thread's columns fit in the
/// caches the threads share. An image whose own columns are more than this is taken in
/// bands of output rows ([`rows_per_chunk`]) instead. The chunks depend on the geometry and
/// the batch alone, never on the thread count.
///
/// ```
/// use cavitation_core::kernels::conv::{COLUMN_BUDGET, Conv2dGeom, images_per_chunk};
/// // A 1-channel 8 x 8 image with a 3 x 3 kernel is 9 · 36 column values: far below the
/// // budget, so a whole group of 8 images is one chunk.
/// let geom = Conv2dGeom::new(1, 4, (8, 8), (3, 3)).unwrap();
/// assert!(9 * 36 * 8 <= COLUMN_BUDGET);
/// assert_eq!(images_per_chunk(&geom, 128), 8);
/// ```
pub const COLUMN_BUDGET: usize = 1 << 17;

/// The number of groups a batch of `batch` images is cut into: `min(batch, CONV_GROUPS)`,
/// at least 1.
///
/// ```
/// use cavitation_core::kernels::conv::groups;
/// assert_eq!((groups(1), groups(9), groups(128)), (1, 9, 16));
/// ```
pub fn groups(batch: usize) -> usize {
    batch.clamp(1, CONV_GROUPS)
}

/// Group `g` of `count` over `batch` images: its first image and its number of images, the
/// images split as evenly as they go.
fn group_range(g: usize, count: usize, batch: usize) -> (usize, usize) {
    let (start, end) = (g * batch / count, (g + 1) * batch / count);
    (start, end - start)
}

/// The number of images each group of a batch of `batch` images takes per GEMM for `geom`:
/// as many as fit in [`COLUMN_BUDGET`], at least 1 and at most the largest group's size.
///
/// ```
/// use cavitation_core::kernels::conv::{COLUMN_BUDGET, Conv2dGeom, images_per_chunk};
/// let geom = Conv2dGeom::with_options(8, 8, (8, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// // 128 images in 16 groups of 8; 72 · 64 column values per image.
/// assert_eq!(images_per_chunk(&geom, 128), 8.min(COLUMN_BUDGET / (72 * 64)));
/// ```
pub fn images_per_chunk(geom: &Conv2dGeom, batch: usize) -> usize {
    let (oh, ow) = geom.out_size();
    let per_image = geom.kernel_len() * oh * ow;
    let largest = batch.div_ceil(groups(batch)).max(1);
    (COLUMN_BUDGET / per_image).clamp(1, largest)
}

/// The fewest windows the widest band [`rows_per_chunk`] may choose holds, before the bands
/// are evened out, so that each band's GEMM stays wide enough for the packed kernels even
/// when the budget alone would give it only a few rows.
const MIN_BAND_WINDOWS: usize = 128;

/// The number of output rows each chunk of an image takes for `geom`: all of them when one
/// image's columns fit in [`COLUMN_BUDGET`]. Else the widest band allowed is as many rows
/// as fit in the budget, raised to at least 128 windows (and one row), and the image is cut
/// into as few bands of at most that many rows as it takes, as even as they go; so a band
/// may exceed the budget for a very wide kernel, or hold fewer than 128 windows after the
/// evening out. Bands of `rows` rows from row 0, the last one shorter when they do not
/// divide the image.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, rows_per_chunk};
/// // 32 channels of 32 x 32 with a 3 x 3 kernel: 288 · 32 column values per output row, so
/// // at most 14 rows fit in the budget, and 32 rows go in 3 bands of 11 (the last of 10).
/// let geom = Conv2dGeom::with_options(32, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert_eq!(rows_per_chunk(&geom), 11);
/// // A small image is taken whole.
/// assert_eq!(rows_per_chunk(&Conv2dGeom::new(1, 4, (8, 8), (3, 3)).unwrap()), 6);
/// ```
pub fn rows_per_chunk(geom: &Conv2dGeom) -> usize {
    let (oh, ow) = geom.out_size();
    let most = (COLUMN_BUDGET / (geom.kernel_len() * ow))
        .max(MIN_BAND_WINDOWS.div_ceil(ow))
        .clamp(1, oh);
    oh.div_ceil(oh.div_ceil(most))
}

/// The windows of one chunk for `batch` images of `geom`: several whole images, or one
/// band of one image's output rows.
fn chunk_windows(geom: &Conv2dGeom, batch: usize) -> usize {
    let (oh, ow) = geom.out_size();
    let rows = rows_per_chunk(geom);
    if rows < oh {
        rows * ow
    } else {
        images_per_chunk(geom, batch) * oh * ow
    }
}

/// The chunks of a group of `len` images, in order, each `(first image, images, first
/// row, rows)` within the group: whole images [`images_per_chunk`] at a time when an
/// image's columns fit in [`COLUMN_BUDGET`], else each image in bands of
/// [`rows_per_chunk`] output rows. They depend on the geometry and the batch alone.
struct Chunks {
    /// The images in the group.
    len: usize,
    /// Images per chunk when whole images are taken.
    per: usize,
    /// Output rows per band, or the whole height when whole images are taken.
    rows: usize,
    /// The output height.
    oh: usize,
    /// The next chunk's first image.
    n: usize,
    /// The next chunk's first row.
    r: usize,
}

impl Chunks {
    /// The chunks of a group of `len` images of a batch of `batch` images of `geom`.
    fn new(geom: &Conv2dGeom, batch: usize, len: usize) -> Chunks {
        Chunks {
            len,
            per: images_per_chunk(geom, batch),
            rows: rows_per_chunk(geom),
            oh: geom.out_size().0,
            n: 0,
            r: 0,
        }
    }
}

impl Iterator for Chunks {
    type Item = (usize, usize, usize, usize);

    /// The next chunk: `(first image, images, first row, rows)`.
    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.len {
            return None;
        }
        if self.rows >= self.oh {
            let (n0, nb) = (self.n, self.per.min(self.len - self.n));
            self.n += nb;
            return Some((n0, nb, 0, self.oh));
        }
        let (n0, r0, nr) = (self.n, self.r, self.rows.min(self.oh - self.r));
        self.r += nr;
        if self.r == self.oh {
            (self.n, self.r) = (self.n + 1, 0);
        }
        Some((n0, 1, r0, nr))
    }
}

/// One worker's scratch for `batch` images of `geom`: a column matrix and a product of one
/// chunk ([`chunk_windows`]), none for a 1 × 1 kernel with stride 1 and no padding.
fn worker_scratch(geom: &Conv2dGeom, batch: usize) -> (usize, usize) {
    if geom.is_pointwise() {
        return (0, 0);
    }
    let windows = chunk_windows(geom, batch);
    (geom.kernel_len() * windows, geom.out_channels() * windows)
}

/// The scratch [`conv2d_forward`] and [`conv2d_backward`] need for `batch` images of
/// `geom` on `threads` threads: one column matrix and one product per thread that takes
/// part.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_scratch};
/// let geom = Conv2dGeom::new(1, 2, (3, 3), (2, 2)).unwrap();
/// assert_eq!(conv2d_scratch(&geom, 1, 4), 4 * 4 + 2 * 4);
/// ```
pub fn conv2d_scratch(geom: &Conv2dGeom, batch: usize, threads: usize) -> usize {
    let (c, p) = worker_scratch(geom, batch);
    (c + p) * workers(batch, threads)
}

/// The threads whose tasks take part for `batch` images: one per group up to `threads`
/// when there are at least as many groups as threads, else one, so that each group's GEMM
/// runs on every thread instead (a small batch would otherwise leave threads idle). Never
/// more than the pool's maximum.
fn workers(batch: usize, threads: usize) -> usize {
    if groups(batch) < threads {
        // Fewer groups than threads: one task at a time, each GEMM on every thread.
        1
    } else {
        threads
            .clamp(1, groups(batch))
            .min(super::pool::MAX_THREADS)
    }
}

/// A pointer that tasks on several threads may hold at once, each using its own part. The
/// convolutions here, max pooling (`super::pooling`) and batch normalisation
/// (`ops::batch_norm`) use it.
pub(crate) struct Parts<T>(pub(crate) *mut T);

// SAFETY: `Parts` is only used by the functions below, by `pooling::max_pool2d` and its
// backward pass and by `ops::batch_norm`, each task of which touches a disjoint part (its
// group's images, its worker's scratch, its block of planes, its channel) of a buffer
// borrowed mutably for the whole call.
unsafe impl<T: Send> Sync for Parts<T> {}

impl<T> Parts<T> {
    /// `len` elements from `start`.
    ///
    /// # Safety
    ///
    /// `start + len` is inside the buffer, and no other live slice overlaps this one.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        // SAFETY: as the caller promises.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// The GEMM a convolution runs: `c = a · b + beta · c` (`beta` 0 or 1). Any closure of that
/// shape that may be shared between threads is one.
///
/// ```
/// use cavitation_core::kernels::conv::ConvGemm;
/// use cavitation_core::kernels::{MatMut, MatRef, reference};
/// fn takes(_: &impl ConvGemm<f64>) {}
/// takes(&|a: MatRef<f64>, b: MatRef<f64>, beta: f64, c: MatMut<f64>| {
///     reference::gemm(1.0, a, b, beta, c)
/// });
/// ```
pub trait ConvGemm<T>: Fn(MatRef<'_, T>, MatRef<'_, T>, T, MatMut<'_, T>) + Sync {}

impl<T, F: Fn(MatRef<'_, T>, MatRef<'_, T>, T, MatMut<'_, T>) + Sync> ConvGemm<T> for F {}

/// The convolution of `batch` images `x` by `weight` (`[out, kernel_len]` row-major, the
/// `[out, in, kh, kw]` weight's own layout), plus `bias`, into `out`, with `gemm` computing
/// the products. The batch is cut into [`groups`] of consecutive images, run as tasks on up
/// to `threads` threads; each group takes its images [`images_per_chunk`] at a time (or, for
/// an image too large for [`COLUMN_BUDGET`], in bands of [`rows_per_chunk`] output rows),
/// copies their windows into its thread's column matrix, multiplies it by the weight into
/// its product (`[out, windows]`) and moves each value to its output image, adding the
/// bias. A 1 × 1 kernel with stride 1 and no padding multiplies each image itself. Each
/// output depends on its own image alone, so the result is the same for any thread count.
/// The tape and `InferenceModel` both run this, so they agree bit for bit at every kernel
/// level whose GEMM gives each value the same bits whatever the product's shape (the
/// library's own kernels); at the Accelerate level, whose `sgemm` chooses its blocking from
/// the shape, a group of another size can round differently, within the GEMM bound.
/// `scratch` holds at least [`conv2d_scratch`] values.
///
/// # Panics
///
/// Panics if any slice is too short or has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_forward, conv2d_scratch};
/// use cavitation_core::kernels::{MatMut, MatRef, reference};
/// let geom = Conv2dGeom::new(1, 1, (3, 3), (2, 2)).unwrap();
/// let x: Vec<f64> = (1..=9).map(f64::from).collect();
/// let mut scratch = vec![0.0; conv2d_scratch(&geom, 1, 1)];
/// let mut out = vec![0.0; 4];
/// let gemm = |a: MatRef<f64>, b: MatRef<f64>, beta: f64, c: MatMut<f64>| {
///     reference::gemm(1.0, a, b, beta, c)
/// };
/// conv2d_forward(&geom, 1, &x, &[1.0; 4], Some(&[0.5]), &mut out, &mut scratch, 1, &gemm);
/// assert_eq!(out, [12.5, 16.5, 24.5, 28.5]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    x: &[T],
    weight: &[T],
    bias: Option<&[T]>,
    out: &mut [T],
    scratch: &mut [T],
    threads: usize,
    gemm: &impl ConvGemm<T>,
) {
    let (oh, ow) = geom.out_size();
    let (windows, kl, cout) = (oh * ow, geom.kernel_len(), geom.out_channels());
    let (in_w, out_w) = (geom.in_width(), geom.out_width());
    check_len("the input", x.len(), batch * in_w);
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output", out.len(), batch * out_w);
    if let Some(b) = bias {
        check_len("the bias", b.len(), cout);
    }
    let (cols_len, prod_len) = worker_scratch(geom, batch);
    let workers = workers(batch, threads);
    assert!(
        scratch.len() >= (cols_len + prod_len) * workers,
        "conv2d: the scratch holds {} values, expected at least {}",
        scratch.len(),
        (cols_len + prod_len) * workers
    );
    let count = groups(batch);
    let wm = MatRef::new(weight, cout, kl, kl, 1).expect("checked length");
    let (outs, scratches) = (Parts(out.as_mut_ptr()), Parts(scratch.as_mut_ptr()));
    super::pool::run(workers, count, &|t| {
        let (first, len) = group_range(t.index, count, batch);
        // SAFETY: groups cover disjoint images, so their outputs are disjoint parts of
        // `out`; each worker runs one task at a time, so its scratch part is its own.
        let (out, own) = unsafe {
            (
                outs.slice(first * out_w, len * out_w),
                scratches.slice(t.worker * (cols_len + prod_len), cols_len + prod_len),
            )
        };
        let (cols, prod) = own.split_at_mut(cols_len);
        let x = &x[first * in_w..][..len * in_w];
        if geom.is_pointwise() {
            for (image, y) in x.chunks_exact(in_w).zip(out.chunks_exact_mut(out_w)) {
                let rhs = MatRef::new(image, kl, windows, windows, 1).expect("checked length");
                let y = MatMut::new(y, cout, windows, windows, 1).expect("checked length");
                gemm(wm, rhs, T::ZERO, y);
            }
        } else {
            for (n0, nb, r0, nr) in Chunks::new(geom, batch, len) {
                let cw = nr * ow;
                let cols_n = nb * cw;
                let cm = &mut cols[..kl * cols_n];
                unroll(geom, nb, &x[n0 * in_w..][..nb * in_w], (r0, nr), cm);
                let pm = &mut prod[..cout * cols_n];
                gemm(
                    wm,
                    MatRef::new(&*cm, kl, cols_n, cols_n, 1).expect("checked length"),
                    T::ZERO,
                    MatMut::new(pm, cout, cols_n, cols_n, 1).expect("checked length"),
                );
                for (k, y) in out[n0 * out_w..][..nb * out_w]
                    .chunks_exact_mut(out_w)
                    .enumerate()
                {
                    for (o, plane) in y.chunks_exact_mut(windows).enumerate() {
                        plane[r0 * ow..][..cw].copy_from_slice(&pm[o * cols_n + k * cw..][..cw]);
                    }
                }
            }
        }
        if let Some(b) = bias {
            for image in out.chunks_exact_mut(out_w) {
                for (plane, &bo) in image.chunks_exact_mut(windows).zip(b) {
                    for v in plane {
                        *v += bo;
                    }
                }
            }
        }
    });
}

/// The input and weight gradients of [`conv2d_forward`] for the output gradient `dy`, group
/// by group as the forward pass cuts the batch, on up to `threads` threads:
///
/// - `dx`, if given, receives the input gradient, `col2im(Wᵀ · dY')` per chunk, added to
///   what is there when its flag is `true` and written over it when `false`;
/// - `dw_parts`, if given, receives one partial weight gradient per group
///   (`groups(batch) · out · kernel_len` values), each `Σ dY' · colsᵀ` over the group's
///   chunks in order, for the caller to add up in group order.
///
/// `dY'` is a chunk's output gradient gathered to `[out, images · windows]`. Every result
/// depends on its group alone, so it is the same for any thread count. `scratch` holds at
/// least [`conv2d_scratch`] values.
///
/// # Panics
///
/// Panics if any slice is too short or has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_backward, conv2d_scratch};
/// use cavitation_core::kernels::{MatMut, MatRef, reference};
/// let geom = Conv2dGeom::new(1, 1, (1, 3), (1, 2)).unwrap();
/// let mut scratch = vec![0.0; conv2d_scratch(&geom, 1, 1)];
/// let (mut dx, mut dw) = (vec![0.0f64; 3], vec![0.0; 2]);
/// let gemm = |a: MatRef<f64>, b: MatRef<f64>, beta: f64, c: MatMut<f64>| {
///     reference::gemm(1.0, a, b, beta, c)
/// };
/// conv2d_backward(&geom, 1, &[1.0, 2.0, 3.0], &[1.0, 1.0], &[1.0, 1.0],
///     Some((&mut dx, false)), Some(&mut dw), &mut scratch, 1, &gemm);
/// assert_eq!((dx, dw), (vec![1.0, 2.0, 1.0], vec![3.0, 5.0]));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    x: &[T],
    weight: &[T],
    dy: &[T],
    dx: Option<(&mut [T], bool)>,
    dw_parts: Option<&mut [T]>,
    scratch: &mut [T],
    threads: usize,
    gemm: &impl ConvGemm<T>,
) {
    let (oh, ow) = geom.out_size();
    let (windows, kl, cout) = (oh * ow, geom.kernel_len(), geom.out_channels());
    let (in_w, out_w) = (geom.in_width(), geom.out_width());
    let count = groups(batch);
    check_len("the input", x.len(), batch * in_w);
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output gradient", dy.len(), batch * out_w);
    let (cols_len, prod_len) = worker_scratch(geom, batch);
    let workers = workers(batch, threads);
    assert!(
        scratch.len() >= (cols_len + prod_len) * workers,
        "conv2d: the scratch holds {} values, expected at least {}",
        scratch.len(),
        (cols_len + prod_len) * workers
    );
    let add = dx.as_ref().is_some_and(|d| d.1);
    let dxs = dx.map(|(d, _)| {
        check_len("the input gradient", d.len(), batch * in_w);
        Parts(d.as_mut_ptr())
    });
    let dws = dw_parts.map(|d| {
        check_len("the weight gradient's parts", d.len(), count * cout * kl);
        Parts(d.as_mut_ptr())
    });
    let wm = MatRef::new(weight, cout, kl, kl, 1).expect("checked length");
    let scratches = Parts(scratch.as_mut_ptr());
    super::pool::run(workers, count, &|t| {
        let (first, len) = group_range(t.index, count, batch);
        // SAFETY: each worker runs one task at a time, so its scratch part is its own.
        let own = unsafe { scratches.slice(t.worker * (cols_len + prod_len), cols_len + prod_len) };
        let (cols, dyp) = own.split_at_mut(cols_len);
        let x = &x[first * in_w..][..len * in_w];
        let dy = &dy[first * out_w..][..len * out_w];
        if let Some(parts) = &dws {
            // SAFETY: group `t.index` alone writes part `t.index`.
            let dw = unsafe { parts.slice(t.index * cout * kl, cout * kl) };
            if len == 0 {
                // An empty batch: its one group adds nothing.
                dw.fill(T::ZERO);
            }
            if geom.is_pointwise() {
                for (n, (image, g)) in x.chunks_exact(in_w).zip(dy.chunks_exact(out_w)).enumerate()
                {
                    let cm = MatRef::new(image, kl, windows, windows, 1).expect("checked length");
                    let gm = MatRef::new(g, cout, windows, windows, 1).expect("checked length");
                    let beta = if n == 0 { T::ZERO } else { T::ONE };
                    gemm(
                        gm,
                        cm.transpose(),
                        beta,
                        MatMut::new(&mut *dw, cout, kl, kl, 1).expect("checked length"),
                    );
                }
            } else {
                for (i, (n0, nb, r0, nr)) in Chunks::new(geom, batch, len).enumerate() {
                    let cols_n = nb * nr * ow;
                    unroll(
                        geom,
                        nb,
                        &x[n0 * in_w..][..nb * in_w],
                        (r0, nr),
                        &mut cols[..kl * cols_n],
                    );
                    gather(
                        &dy[n0 * out_w..][..nb * out_w],
                        (windows, ow, r0, nr),
                        cout,
                        &mut dyp[..cout * cols_n],
                    );
                    let gm = MatRef::new(&dyp[..cout * cols_n], cout, cols_n, cols_n, 1)
                        .expect("checked length");
                    let cm = MatRef::new(&cols[..kl * cols_n], kl, cols_n, cols_n, 1)
                        .expect("checked length");
                    let beta = if i == 0 { T::ZERO } else { T::ONE };
                    gemm(
                        gm,
                        cm.transpose(),
                        beta,
                        MatMut::new(&mut *dw, cout, kl, kl, 1).expect("checked length"),
                    );
                }
            }
        }
        if let Some(parts) = &dxs {
            // SAFETY: groups cover disjoint images, so their input gradients are disjoint.
            let dx = unsafe { parts.slice(first * in_w, len * in_w) };
            let wt = wm.transpose();
            let beta = if add { T::ONE } else { T::ZERO };
            if geom.is_pointwise() {
                for (g, d) in dy.chunks_exact(out_w).zip(dx.chunks_exact_mut(in_w)) {
                    let gm = MatRef::new(g, cout, windows, windows, 1).expect("checked length");
                    gemm(
                        wt,
                        gm,
                        beta,
                        MatMut::new(d, kl, windows, windows, 1).expect("checked length"),
                    );
                }
            } else {
                if !add {
                    dx.fill(T::ZERO);
                }
                for (n0, nb, r0, nr) in Chunks::new(geom, batch, len) {
                    let cols_n = nb * nr * ow;
                    gather(
                        &dy[n0 * out_w..][..nb * out_w],
                        (windows, ow, r0, nr),
                        cout,
                        &mut dyp[..cout * cols_n],
                    );
                    let gm = MatRef::new(&dyp[..cout * cols_n], cout, cols_n, cols_n, 1)
                        .expect("checked length");
                    let dcols = &mut cols[..kl * cols_n];
                    gemm(
                        wt,
                        gm,
                        T::ZERO,
                        MatMut::new(&mut *dcols, kl, cols_n, cols_n, 1).expect("checked length"),
                    );
                    roll_back(geom, nb, dcols, (r0, nr), &mut dx[n0 * in_w..][..nb * in_w]);
                }
            }
        }
    });
}

/// Whether [`conv3x3_direct`] runs `geom` at `level`: a 3 × 3 kernel with stride 1, padding
/// 1 and dilation 1, output channels a multiple of 4, a width that is a multiple of the
/// level's vector width (4 at SSE4.2 and NEON, 8 at AVX2), at a level with one of those
/// kernels on this CPU. Every other convolution runs through im2col and the GEMM.
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::conv::{Conv2dGeom, direct_runs};
/// let g = Conv2dGeom::with_options(32, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert!(!direct_runs(Level::Reference, &g));
/// ```
pub fn direct_runs(level: Level, geom: &Conv2dGeom) -> bool {
    let lanes = match level {
        Level::Sse42 | Level::Neon => 4,
        Level::Avx2Fma => 8,
        _ => return false,
    };
    super::supported(level)
        && geom.kernel == (3, 3)
        && geom.stride == (1, 1)
        && geom.padding == (1, 1)
        && geom.dilation == (1, 1)
        && geom.out_channels.is_multiple_of(4)
        && geom.input.1.is_multiple_of(lanes)
}

/// Whether the input gradient of `geom` runs through [`conv3x3_direct_input_grad`] at
/// `level`: as [`direct_runs`], with the input channels (the gradient's output channels) a
/// multiple of 4 instead.
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::conv::{Conv2dGeom, direct_input_grad_runs};
/// let g = Conv2dGeom::with_options(3, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert!(!direct_input_grad_runs(Level::Avx2Fma, &g)); // 3 input channels
/// ```
pub fn direct_input_grad_runs(level: Level, geom: &Conv2dGeom) -> bool {
    let flipped = Conv2dGeom {
        in_channels: geom.out_channels,
        out_channels: geom.in_channels,
        ..*geom
    };
    direct_runs(level, &flipped)
}

/// The scratch [`conv3x3_direct`] and [`conv3x3_direct_input_grad`] need for `geom` on
/// `threads` threads at any level: the weights in blocks of four channels (each stored four
/// times over at SSE4.2), and one zero-padded image per worker.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, direct_scratch};
/// let g = Conv2dGeom::with_options(4, 8, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert_eq!(direct_scratch(&g, 2), 4 * 8 * 9 * 4 + 2 * 8 * 6 * 6);
/// ```
pub fn direct_scratch(geom: &Conv2dGeom, threads: usize) -> usize {
    let (h, w) = geom.input;
    let channels = geom.in_channels.max(geom.out_channels);
    4 * geom.weight_len()
        + threads.clamp(1, super::pool::MAX_THREADS) * channels * (h + 2) * (w + 2)
}

/// The direct 3 × 3 convolution of `batch` images `x` by `weight` (`[out, in, 3, 3]`), plus
/// `bias`, into `out`, with stride 1 and padding 1, at kernel level `level`, for a geometry
/// [`direct_runs`] accepts. Each image is a task on up to `threads` threads: it is copied
/// into a zero-padded buffer, and every output is summed from zero over the input channels
/// in order and within each over the kernel's rows and columns, then the bias is added. At
/// SSE4.2 each step is a multiply and then an add; at AVX2 and NEON it is one fused
/// multiply-add. [`conv3x3_direct_twin`] sums in the same order and gives the same bits.
/// The result does not depend on `threads`. `scratch` holds at least [`direct_scratch`]
/// values.
///
/// # Panics
///
/// Panics if [`direct_runs`] refuses `geom` at `level`, or if a slice has the wrong length,
/// naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv3x3_direct, conv3x3_direct_twin, direct_runs, direct_scratch};
/// use cavitation_core::kernels::level;
/// let g = Conv2dGeom::with_options(2, 4, (3, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// if direct_runs(level(), &g) {
///     let x: Vec<f32> = (0..g.in_width()).map(|i| i as f32 / 7.0).collect();
///     let w: Vec<f32> = (0..g.weight_len()).map(|i| 1.0 - i as f32 / 50.0).collect();
///     let (mut fast, mut twin) = (vec![0.0; g.out_width()], vec![0.0; g.out_width()]);
///     let mut scratch = vec![0.0; direct_scratch(&g, 1)];
///     conv3x3_direct(level(), &g, 1, &x, &w, None, &mut fast, &mut scratch, 1);
///     conv3x3_direct_twin(level(), &g, 1, &x, &w, None, &mut twin, false);
///     assert_eq!(fast, twin);
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_direct(
    level: Level,
    geom: &Conv2dGeom,
    batch: usize,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    scratch: &mut [f32],
    threads: usize,
) {
    assert!(
        direct_runs(level, geom),
        "conv3x3_direct: {geom:?} does not run directly at {level:?}"
    );
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output", out.len(), batch * geom.out_width());
    if let Some(b) = bias {
        check_len("the bias", b.len(), geom.out_channels);
    }
    let (cin, cout) = (geom.in_channels, geom.out_channels);
    direct_run(
        level,
        (cin, cout, geom.input),
        batch,
        x,
        |o, c, k| weight[(o * cin + c) * 9 + k],
        bias,
        (out, false),
        scratch,
        threads,
    );
}

/// The input gradient of a convolution [`direct_runs`] accepts, for a geometry
/// [`direct_input_grad_runs`] accepts: `batch` output gradients `dy` convolved with
/// `weight` transposed and flipped (`dx[c] = Σ_o dy[o] ⋆ w[o][c]` turned by 180°), stride 1
/// and padding 1, into `dx`, or added to `dx` with `add`. As [`conv3x3_direct`] in every
/// other way, with the sums over the output channels in order.
///
/// # Panics
///
/// Panics if [`direct_input_grad_runs`] refuses `geom` at `level`, or if a slice has the
/// wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv3x3_direct_input_grad, direct_input_grad_runs, direct_scratch};
/// use cavitation_core::kernels::level;
/// // One channel each way, a kernel with 1 only at its centre: the gradient is dy itself.
/// let g = Conv2dGeom::with_options(4, 4, (1, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// if direct_input_grad_runs(level(), &g) {
///     let mut w = vec![0.0f32; g.weight_len()];
///     for c in 0..4 { w[(c * 4 + c) * 9 + 4] = 1.0; }
///     let dy: Vec<f32> = (0..g.out_width()).map(|i| i as f32).collect();
///     let mut dx = vec![0.0; g.in_width()];
///     let mut scratch = vec![0.0; direct_scratch(&g, 1)];
///     conv3x3_direct_input_grad(level(), &g, 1, &dy, &w, &mut dx, false, &mut scratch, 1);
///     assert_eq!(dx, dy);
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_direct_input_grad(
    level: Level,
    geom: &Conv2dGeom,
    batch: usize,
    dy: &[f32],
    weight: &[f32],
    dx: &mut [f32],
    add: bool,
    scratch: &mut [f32],
    threads: usize,
) {
    assert!(
        direct_input_grad_runs(level, geom),
        "conv3x3_direct_input_grad: {geom:?} does not run directly at {level:?}"
    );
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    let (cin, cout) = (geom.in_channels, geom.out_channels);
    direct_run(
        level,
        (cout, cin, geom.input),
        batch,
        dy,
        |c, o, k| weight[(o * cin + c) * 9 + 8 - k],
        None,
        (dx, add),
        scratch,
        threads,
    );
}

/// Runs the direct kernel of `level` over `batch` images `src` of `cin` planes of `(h, w)`
/// into `dst` (`cout` planes each), with weight `(o, c, k)` (tap `k = 3·ky + kx`) from
/// `weight`, after checking the scratch.
#[allow(clippy::too_many_arguments)]
fn direct_run(
    level: Level,
    (cin, cout, (h, w)): (usize, usize, (usize, usize)),
    batch: usize,
    src: &[f32],
    weight: impl Fn(usize, usize, usize) -> f32,
    bias: Option<&[f32]>,
    (dst, add): (&mut [f32], bool),
    scratch: &mut [f32],
    threads: usize,
) {
    let (wp, plane) = (w + 2, (h + 2) * (w + 2));
    let workers = threads.min(batch).clamp(1, super::pool::MAX_THREADS);
    let pad_len = cin * plane;
    // SSE4.2 reads each weight already broadcast to four lanes: a broadcast from memory
    // costs it a shuffle, which on cores with one four-wide multiply per cycle costs as much
    // as a multiply. AVX2 and NEON broadcast in the load itself.
    let rep = if level == Level::Sse42 { 4 } else { 1 };
    let weights_len = cin * cout * 9 * rep;
    let need = weights_len + workers * pad_len;
    assert!(
        scratch.len() >= need,
        "conv3x3_direct: the scratch holds {} values, expected at least {need}",
        scratch.len()
    );
    let (packed, pads) = scratch.split_at_mut(weights_len);
    for ob in 0..cout / 4 {
        for c in 0..cin {
            for k in 0..9 {
                for o in 0..4 {
                    let at = (((ob * cin + c) * 9 + k) * 4 + o) * rep;
                    packed[at..at + rep].fill(weight(ob * 4 + o, c, k));
                }
            }
        }
    }
    let packed = &*packed;
    let (in_w, out_w) = (cin * h * w, cout * h * w);
    let (outs, pads) = (Parts(dst.as_mut_ptr()), Parts(pads.as_mut_ptr()));
    super::pool::run(workers, batch, &|t| {
        // SAFETY: each worker runs one task at a time, so its padded image is its own; image
        // `t.index`'s outputs are its own part of `dst`.
        let (pad, out) = unsafe {
            (
                pads.slice(t.worker * pad_len, pad_len),
                outs.slice(t.index * out_w, out_w),
            )
        };
        let image = &src[t.index * in_w..][..in_w];
        for (c, p) in pad.chunks_exact_mut(plane).enumerate() {
            p[..wp].fill(0.0);
            p[(h + 1) * wp..].fill(0.0);
            for y in 0..h {
                let row = &mut p[(y + 1) * wp..][..wp];
                row[0] = 0.0;
                row[w + 1] = 0.0;
                row[1..=w].copy_from_slice(&image[(c * h + y) * w..][..w]);
            }
        }
        let job = direct::Job {
            pad: pad.as_ptr(),
            weights: packed.as_ptr(),
            bias: bias.map(<[f32]>::as_ptr),
            out: out.as_mut_ptr(),
            add,
            cin,
            cout,
            h,
            w,
        };
        // SAFETY: `pad` holds `cin` padded planes, `packed` the weights in blocks of four
        // channels, `out` the image's `cout` output planes and `bias` `cout` values;
        // `direct_runs` (checked by the caller) makes `w` a multiple of the level's lanes and
        // `cout` of 4, and says this CPU supports `level`.
        unsafe {
            match level {
                #[cfg(target_arch = "x86_64")]
                Level::Sse42 => direct::direct_sse(&job),
                #[cfg(target_arch = "x86_64")]
                Level::Avx2Fma => direct::direct_avx2(&job),
                #[cfg(target_arch = "aarch64")]
                Level::Neon => direct::direct_neon(&job),
                _ => unreachable!("direct_runs accepts only SSE4.2, AVX2 and NEON"),
            }
        }
    });
}

/// The scalar twin of [`conv3x3_direct`] (and, given the transposed and flipped weight, of
/// [`conv3x3_direct_input_grad`]): the same sums in the same order, each step `acc + x · w`
/// rounded twice at SSE4.2 and once (`mul_add`) at AVX2 and NEON, the bias then added, or
/// with `add` the sum added to what `out` holds. Runs any 3 × 3, stride-1, padding-1,
/// dilation-1 geometry, one image after another.
///
/// # Panics
///
/// Panics if `geom` is not such a geometry, or if a slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv3x3_direct_twin};
/// // One 1 x 1 image, a kernel of 2s: only the centre tap is inside the image.
/// let g = Conv2dGeom::with_options(1, 1, (1, 1), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// let mut out = [0.0f32];
/// conv3x3_direct_twin(Level::Sse42, &g, 1, &[3.0], &[2.0; 9], Some(&[1.0]), &mut out, false);
/// assert_eq!(out, [7.0]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_direct_twin(
    level: Level,
    geom: &Conv2dGeom,
    batch: usize,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    add: bool,
) {
    assert!(
        geom.kernel == (3, 3)
            && geom.stride == (1, 1)
            && geom.padding == (1, 1)
            && geom.dilation == (1, 1),
        "conv3x3_direct_twin: {geom:?} is not a 3 x 3, stride-1, padding-1 convolution"
    );
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output", out.len(), batch * geom.out_width());
    let fused = matches!(level, Level::Avx2Fma | Level::Neon);
    let (cin, cout, (h, w)) = (geom.in_channels, geom.out_channels, geom.input);
    for n in 0..batch {
        let image = &x[n * cin * h * w..][..cin * h * w];
        for o in 0..cout {
            for y in 0..h {
                for xo in 0..w {
                    let mut acc = 0.0f32;
                    for c in 0..cin {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let (iy, ix) = (y + ky, xo + kx);
                                let v = if iy < 1 || iy > h || ix < 1 || ix > w {
                                    0.0
                                } else {
                                    image[(c * h + iy - 1) * w + ix - 1]
                                };
                                let wv = weight[((o * cin + c) * 3 + ky) * 3 + kx];
                                acc = if fused {
                                    v.mul_add(wv, acc)
                                } else {
                                    acc + v * wv
                                };
                            }
                        }
                    }
                    if let Some(b) = bias {
                        acc += b[o];
                    }
                    let at = &mut out[((n * cout + o) * h + y) * w + xo];
                    *at = if add { *at + acc } else { acc };
                }
            }
        }
    }
}

/// The output channels one task of [`conv3x3_direct_weight_grad`] takes: it pads each input
/// plane once for all of them.
const WEIGHT_GRAD_BLOCK: usize = 8;

/// Whether the weight gradient of `geom` runs through [`conv3x3_direct_weight_grad`] at
/// `level`: a 3 × 3 kernel with stride 1, padding 1 and dilation 1, a width that is a
/// multiple of the level's vector width (4 at SSE4.2 and NEON, 8 at AVX2), at a level with
/// one of those kernels on this CPU. Any channel counts.
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::conv::{Conv2dGeom, direct_weight_grad_runs};
/// let g = Conv2dGeom::with_options(3, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert!(!direct_weight_grad_runs(Level::Reference, &g));
/// ```
pub fn direct_weight_grad_runs(level: Level, geom: &Conv2dGeom) -> bool {
    let lanes = match level {
        Level::Sse42 | Level::Neon => 4,
        Level::Avx2Fma => 8,
        _ => return false,
    };
    super::supported(level)
        && geom.kernel == (3, 3)
        && geom.stride == (1, 1)
        && geom.padding == (1, 1)
        && geom.dilation == (1, 1)
        && geom.input.1.is_multiple_of(lanes)
}

/// The scratch [`conv3x3_direct_weight_grad`] needs for `geom` on `threads` threads: for
/// each worker, one zero-padded input plane and the running totals of a block of up to
/// eight output channels.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, direct_weight_grad_scratch};
/// let g = Conv2dGeom::with_options(4, 16, (4, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert_eq!(direct_weight_grad_scratch(&g, 2), 2 * (6 * 6 + 8 * 4 * 9));
/// ```
pub fn direct_weight_grad_scratch(geom: &Conv2dGeom, threads: usize) -> usize {
    let (h, w) = geom.input;
    let block = WEIGHT_GRAD_BLOCK.min(geom.out_channels);
    threads.clamp(1, super::pool::MAX_THREADS) * ((h + 2) * (w + 2) + block * geom.in_channels * 9)
}

/// The weight gradient of a convolution [`direct_weight_grad_runs`] accepts: for each
/// output channel `o`, input channel `c` and tap `(ky, kx)`,
/// `dw[o][c][ky][kx] = Σₙ Σ_{y, x} dy[n][o][y][x] · x[n][c][y + ky − 1][x + kx − 1]` (0 outside
/// the image), written into `dw` (`[out, in, 3, 3]`), or added to it with `add`, at kernel
/// level `level`.
///
/// Each block of up to eight output channels is a task on up to `threads` threads, which
/// owns that block's rows of `dw`. For each image in order and each input channel, it
/// copies the plane into a zero-padded buffer and adds the image's sum for each output
/// channel and tap to a running total that starts at zero; each image's sum runs one
/// accumulator per vector lane and adds the lanes as a tree (see the kernels' module). At
/// SSE4.2 each step is a multiply and then an add; at AVX2 and NEON it is one fused
/// multiply-add. [`conv3x3_direct_weight_grad_twin`] sums in the same order and gives the
/// same bits, and the result does not depend on `threads`. With `add`, the total is then
/// added to what `dw` holds. `scratch` holds at least [`direct_weight_grad_scratch`] values.
///
/// # Panics
///
/// Panics if [`direct_weight_grad_runs`] refuses `geom` at `level`, or if a slice has the
/// wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{
///     Conv2dGeom, conv3x3_direct_weight_grad, conv3x3_direct_weight_grad_twin,
///     direct_weight_grad_runs, direct_weight_grad_scratch,
/// };
/// use cavitation_core::kernels::level;
/// let g = Conv2dGeom::with_options(2, 3, (4, 8), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// if direct_weight_grad_runs(level(), &g) {
///     let x: Vec<f32> = (0..2 * g.in_width()).map(|i| i as f32 / 9.0 - 3.0).collect();
///     let dy: Vec<f32> = (0..2 * g.out_width()).map(|i| 1.0 - i as f32 / 40.0).collect();
///     let (mut fast, mut twin) = (vec![0.0; g.weight_len()], vec![0.0; g.weight_len()]);
///     let mut scratch = vec![0.0; direct_weight_grad_scratch(&g, 2)];
///     conv3x3_direct_weight_grad(level(), &g, 2, &x, &dy, &mut fast, false, &mut scratch, 2);
///     conv3x3_direct_weight_grad_twin(level(), &g, 2, &x, &dy, &mut twin, false);
///     assert_eq!(fast, twin);
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_direct_weight_grad(
    level: Level,
    geom: &Conv2dGeom,
    batch: usize,
    x: &[f32],
    dy: &[f32],
    dw: &mut [f32],
    add: bool,
    scratch: &mut [f32],
    threads: usize,
) {
    assert!(
        direct_weight_grad_runs(level, geom),
        "conv3x3_direct_weight_grad: {geom:?} does not run directly at {level:?}"
    );
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the weight gradient", dw.len(), geom.weight_len());
    let (cin, cout, (h, w)) = (geom.in_channels, geom.out_channels, geom.input);
    let (wp, plane) = (w + 2, (h + 2) * (w + 2));
    let block = WEIGHT_GRAD_BLOCK.min(cout);
    let tasks = cout.div_ceil(block);
    let workers = threads.min(tasks).clamp(1, super::pool::MAX_THREADS);
    let per_worker = plane + block * cin * 9;
    let need = workers * per_worker;
    assert!(
        scratch.len() >= need,
        "conv3x3_direct_weight_grad: the scratch holds {} values, expected at least {need}",
        scratch.len()
    );
    let (in_w, out_w) = (cin * h * w, cout * h * w);
    let (rows, parts) = (Parts(dw.as_mut_ptr()), Parts(scratch.as_mut_ptr()));
    super::pool::run(workers, tasks, &|t| {
        let (o0, o1) = (t.index * block, ((t.index + 1) * block).min(cout));
        let len = (o1 - o0) * cin * 9;
        // SAFETY: each worker runs one task at a time, so its scratch is its own; the
        // block's rows of `dw` belong to this task alone.
        let (mine, dst) = unsafe {
            (
                parts.slice(t.worker * per_worker, per_worker),
                rows.slice(o0 * cin * 9, len),
            )
        };
        let (pad, totals) = mine.split_at_mut(plane);
        let totals = &mut totals[..len];
        pad.fill(0.0);
        totals.fill(0.0);
        for n in 0..batch {
            let image = &x[n * in_w..][..in_w];
            for c in 0..cin {
                for y in 0..h {
                    pad[(y + 1) * wp + 1..][..w].copy_from_slice(&image[(c * h + y) * w..][..w]);
                }
                let job = direct::WeightJob {
                    pad: pad.as_ptr(),
                    dy: dy[n * out_w..].as_ptr(),
                    out: totals.as_mut_ptr(),
                    c,
                    cin,
                    o0,
                    o1,
                    h,
                    w,
                };
                // SAFETY: `pad` holds one padded plane, `dy` the image's `cout` output
                // planes and `totals` the block's `[o1 - o0][cin][9]` totals;
                // `direct_weight_grad_runs` (checked above) makes `w` a multiple of the
                // level's lanes and says this CPU supports `level`.
                unsafe {
                    match level {
                        #[cfg(target_arch = "x86_64")]
                        Level::Sse42 => direct::weight_grad_sse(&job),
                        #[cfg(target_arch = "x86_64")]
                        Level::Avx2Fma => direct::weight_grad_avx2(&job),
                        #[cfg(target_arch = "aarch64")]
                        Level::Neon => direct::weight_grad_neon(&job),
                        _ => unreachable!(
                            "direct_weight_grad_runs accepts only SSE4.2, AVX2 and NEON"
                        ),
                    }
                }
            }
        }
        for (d, &s) in dst.iter_mut().zip(totals.iter()) {
            *d = if add { *d + s } else { s };
        }
    });
}

/// The scalar twin of [`conv3x3_direct_weight_grad`]: the same sums in the same order (one
/// accumulator per lane of `level`'s vector width, each step `acc + dy · x` rounded twice
/// at SSE4.2 and once (`mul_add`) at AVX2 and NEON, the lanes added as a tree, image after
/// image into a total from zero), then written into `dw` or with `add` added to it. Runs
/// any 3 × 3, stride-1, padding-1, dilation-1 geometry whose width is a multiple of the
/// lane count.
///
/// # Panics
///
/// Panics if `geom` is not such a geometry, if `level` is not SSE4.2, AVX2 or NEON, or if
/// a slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::Level;
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv3x3_direct_weight_grad_twin};
/// // One 1 x 4 image of ones and a gradient of ones: the centre tap sees all four values,
/// // the side taps three, the rows above and below none.
/// let g = Conv2dGeom::with_options(1, 1, (1, 4), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// let mut dw = [0.0f32; 9];
/// conv3x3_direct_weight_grad_twin(Level::Sse42, &g, 1, &[1.0; 4], &[1.0; 4], &mut dw, false);
/// assert_eq!(dw, [0.0, 0.0, 0.0, 3.0, 4.0, 3.0, 0.0, 0.0, 0.0]);
/// ```
pub fn conv3x3_direct_weight_grad_twin(
    level: Level,
    geom: &Conv2dGeom,
    batch: usize,
    x: &[f32],
    dy: &[f32],
    dw: &mut [f32],
    add: bool,
) {
    let lanes = match level {
        Level::Sse42 | Level::Neon => 4,
        Level::Avx2Fma => 8,
        other => panic!("conv3x3_direct_weight_grad_twin: no direct kernel at {other:?}"),
    };
    assert!(
        geom.kernel == (3, 3)
            && geom.stride == (1, 1)
            && geom.padding == (1, 1)
            && geom.dilation == (1, 1)
            && geom.input.1.is_multiple_of(lanes),
        "conv3x3_direct_weight_grad_twin: {geom:?} is not a 3 x 3, stride-1, padding-1 \
         convolution {lanes} lanes wide"
    );
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the weight gradient", dw.len(), geom.weight_len());
    let fused = lanes == 8 || level == Level::Neon;
    let (cin, cout, (h, w)) = (geom.in_channels, geom.out_channels, geom.input);
    let input = |n: usize, c: usize, iy: usize, ix: usize| {
        if iy < 1 || iy > h || ix < 1 || ix > w {
            0.0
        } else {
            x[((n * cin + c) * h + iy - 1) * w + ix - 1]
        }
    };
    for o in 0..cout {
        for c in 0..cin {
            for t in 0..9 {
                let (ky, kx) = (t / 3, t % 3);
                let mut total = 0.0f32;
                for n in 0..batch {
                    let d = &dy[(n * cout + o) * h * w..][..h * w];
                    let mut acc = vec![0.0f32; lanes];
                    for y in 0..h {
                        for x0 in (0..w).step_by(lanes) {
                            for (i, a) in acc.iter_mut().enumerate() {
                                let (g, v) = (d[y * w + x0 + i], input(n, c, y + ky, x0 + i + kx));
                                *a = if fused { g.mul_add(v, *a) } else { *a + g * v };
                            }
                        }
                    }
                    total += direct::tree_sum(&acc);
                }
                let at = &mut dw[(o * cin + c) * 9 + t];
                *at = if add { *at + total } else { total };
            }
        }
    }
}

/// Gathers output rows `r0..r0 + nr` of the output gradient of `dy.len() / (out ·
/// windows)` consecutive images, each `[out, windows]` with rows of `ow` windows, into `dst`
/// as `[out, images · nr · ow]`: channel `o` of every image side by side, image after image.
fn gather<T: Real>(
    dy: &[T],
    (windows, ow, r0, nr): (usize, usize, usize, usize),
    cout: usize,
    dst: &mut [T],
) {
    let images = dy.len() / (cout * windows);
    let cw = nr * ow;
    let cols_n = images * cw;
    for (k, image) in dy.chunks_exact(cout * windows).enumerate() {
        for (o, plane) in image.chunks_exact(windows).enumerate() {
            dst[o * cols_n + k * cw..][..cw].copy_from_slice(&plane[r0 * ow..][..cw]);
        }
    }
}

/// The direct convolution of `batch` images `x` by `weight` (`[out, in, kh, kw]`), plus
/// `bias` (`[out]`) if given, into `out` (`batch` output images): for each output value, in
/// `T`, `acc = 0; for c, i, j in order { acc = acc + w[o][c][i][j] · x[..] }` over the taps
/// inside the image, then `acc + bias[o]`. The reference twin of the fast path.
///
/// # Panics
///
/// Panics if any slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_reference};
/// let geom = Conv2dGeom::new(1, 1, (1, 3), (1, 1)).unwrap();
/// let mut out = vec![0.0f32; 3];
/// conv2d_reference(&geom, 1, &[1.0, 2.0, 3.0], &[2.0], Some(&[0.5]), &mut out);
/// assert_eq!(out, [2.5, 4.5, 6.5]);
/// ```
pub fn conv2d_reference<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    x: &[T],
    weight: &[T],
    bias: Option<&[T]>,
    out: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dw)) = (geom.stride, geom.padding, geom.dilation);
    let cin = geom.in_channels;
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the weight", weight.len(), geom.weight_len());
    if let Some(b) = bias {
        check_len("the bias", b.len(), geom.out_channels);
    }
    check_len("the output", out.len(), batch * geom.out_width());
    for n in 0..batch {
        for o in 0..geom.out_channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = T::ZERO;
                    for c in 0..cin {
                        for i in 0..kh {
                            let Some(y) = Conv2dGeom::source(oy, i, sh, ph, dh, h) else {
                                continue;
                            };
                            for j in 0..kw {
                                let Some(xx) = Conv2dGeom::source(ox, j, sw, pw, dw, w) else {
                                    continue;
                                };
                                let wv = weight[((o * cin + c) * kh + i) * kw + j];
                                acc += wv * x[((n * cin + c) * h + y) * w + xx];
                            }
                        }
                    }
                    let b = bias.map_or(T::ZERO, |b| b[o]);
                    out[((n * geom.out_channels + o) * oh + oy) * ow + ox] = acc + b;
                }
            }
        }
    }
}

/// The gradients of [`conv2d_reference`] for the output gradient `dy`, by direct loops:
/// `dx` (input), `dw` (weight) and `db` (bias), each overwritten. `dw[o][c][i][j]` sums
/// `dy · x` over images then output positions in order, `db[o]` sums `dy` the same way, and
/// each `dx` value gathers its contributions by output channel, then kernel tap, then output
/// position.
///
/// # Panics
///
/// Panics if any slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_backward_reference};
/// let geom = Conv2dGeom::new(1, 1, (1, 3), (1, 2)).unwrap();
/// let (mut dx, mut dw, mut db) = (vec![0.0f64; 3], vec![0.0; 2], vec![0.0; 1]);
/// conv2d_backward_reference(&geom, 1, &[1.0, 2.0, 3.0], &[1.0, 1.0], &[1.0, 1.0],
///     &mut dx, &mut dw, &mut db);
/// assert_eq!((dx, dw, db), (vec![1.0, 2.0, 1.0], vec![3.0, 5.0], vec![2.0]));
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_reference<T: Real>(
    geom: &Conv2dGeom,
    batch: usize,
    x: &[T],
    weight: &[T],
    dy: &[T],
    dx: &mut [T],
    dw: &mut [T],
    db: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dwl)) = (geom.stride, geom.padding, geom.dilation);
    let (cin, cout) = (geom.in_channels, geom.out_channels);
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    check_len("the weight gradient", dw.len(), geom.weight_len());
    check_len("the bias gradient", db.len(), cout);
    let at = |n: usize, o: usize, oy: usize, ox: usize| ((n * cout + o) * oh + oy) * ow + ox;
    for (o, d) in db.iter_mut().enumerate() {
        let mut acc = T::ZERO;
        for n in 0..batch {
            for p in 0..oh * ow {
                acc += dy[at(n, o, 0, 0) + p];
            }
        }
        *d = acc;
    }
    for o in 0..cout {
        for c in 0..cin {
            for i in 0..kh {
                for j in 0..kw {
                    let mut acc = T::ZERO;
                    for n in 0..batch {
                        for oy in 0..oh {
                            let Some(y) = Conv2dGeom::source(oy, i, sh, ph, dh, h) else {
                                continue;
                            };
                            for ox in 0..ow {
                                let Some(xx) = Conv2dGeom::source(ox, j, sw, pw, dwl, w) else {
                                    continue;
                                };
                                acc += dy[at(n, o, oy, ox)] * x[((n * cin + c) * h + y) * w + xx];
                            }
                        }
                    }
                    dw[((o * cin + c) * kh + i) * kw + j] = acc;
                }
            }
        }
    }
    dx.fill(T::ZERO);
    for n in 0..batch {
        for c in 0..cin {
            for o in 0..cout {
                for i in 0..kh {
                    for j in 0..kw {
                        let wv = weight[((o * cin + c) * kh + i) * kw + j];
                        for oy in 0..oh {
                            let Some(y) = Conv2dGeom::source(oy, i, sh, ph, dh, h) else {
                                continue;
                            };
                            for ox in 0..ow {
                                let Some(xx) = Conv2dGeom::source(ox, j, sw, pw, dwl, w) else {
                                    continue;
                                };
                                let d = &mut dx[((n * cin + c) * h + y) * w + xx];
                                *d += wv * dy[at(n, o, oy, ox)];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Tests for the convolution geometry, the column layout and the direct loops.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    /// Output sizes for stride, padding, dilation and non-square cases, worked out from
    /// PyTorch's formula by hand.
    #[test]
    fn out_size_by_hand() {
        let g = |input, kernel, stride, padding, dilation| {
            Conv2dGeom::with_options(2, 3, input, kernel, stride, padding, dilation)
                .map(|g| g.out_size())
        };
        assert_eq!(
            g((32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(),
            (32, 32)
        );
        assert_eq!(
            g((32, 32), (3, 3), (2, 2), (1, 1), (1, 1)).unwrap(),
            (16, 16)
        );
        assert_eq!(
            g((28, 28), (5, 5), (1, 1), (2, 2), (1, 1)).unwrap(),
            (28, 28)
        );
        assert_eq!(g((10, 7), (3, 2), (2, 3), (0, 1), (1, 1)).unwrap(), (4, 3));
        assert_eq!(g((9, 9), (3, 3), (1, 1), (0, 0), (3, 3)).unwrap(), (3, 3));
        assert_eq!(g((3, 3), (3, 3), (1, 1), (0, 0), (1, 1)).unwrap(), (1, 1));
        assert_eq!(g((1, 1), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(), (1, 1));
        assert_eq!(g((5, 5), (2, 2), (7, 7), (0, 0), (1, 1)).unwrap(), (1, 1));
    }

    /// A geometry with no window, or a zero count, size, stride or dilation, is refused,
    /// naming the sizes.
    #[test]
    fn bad_geometry_is_refused() {
        let e = Conv2dGeom::new(1, 1, (4, 4), (5, 5))
            .unwrap_err()
            .to_string();
        assert!(e.contains("does not fit an image of (4, 4)"), "{e}");
        let e = Conv2dGeom::new(1, 1, (7, 7), (3, 3))
            .unwrap()
            .with_dilation((4, 4))
            .unwrap_err()
            .to_string();
        assert!(e.contains("dilation (4, 4)"), "{e}");
        assert!(Conv2dGeom::new(0, 1, (4, 4), (3, 3)).is_err());
        assert!(Conv2dGeom::new(1, 0, (4, 4), (3, 3)).is_err());
        assert!(Conv2dGeom::new(1, 1, (0, 4), (1, 1)).is_err());
        assert!(Conv2dGeom::new(1, 1, (4, 4), (0, 1)).is_err());
        let g = Conv2dGeom::new(1, 1, (4, 4), (3, 3)).unwrap();
        assert!(g.with_stride((0, 1)).is_err());
        assert!(g.with_dilation((1, 0)).is_err());
        assert!(g.with_padding((usize::MAX, 0)).is_err());
    }

    /// The words round trip.
    #[test]
    fn words_round_trip() {
        let g = Conv2dGeom::new(3, 5, (9, 8), (3, 2))
            .unwrap()
            .with_stride((2, 1))
            .unwrap()
            .with_padding((1, 0))
            .unwrap()
            .with_dilation((1, 2))
            .unwrap();
        assert_eq!(Conv2dGeom::from_words(&g.to_words()), g);
    }

    /// Seeded geometries covering stride, padding, dilation and non-square shapes.
    pub(crate) fn geometries() -> Vec<Conv2dGeom> {
        let mut out = Vec::new();
        let cases = [
            (2, 3, (5, 5), (3, 3), (1, 1), (0, 0), (1, 1)),
            (1, 2, (6, 7), (3, 2), (2, 1), (1, 0), (1, 1)),
            (3, 2, (7, 7), (3, 3), (1, 1), (2, 2), (2, 2)),
            (2, 2, (4, 6), (1, 1), (1, 1), (0, 0), (1, 1)),
            (2, 1, (5, 4), (2, 3), (2, 2), (1, 1), (1, 1)),
            (1, 1, (3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
        ];
        for (cin, cout, input, kernel, stride, padding, dilation) in cases {
            out.push(
                Conv2dGeom::with_options(cin, cout, input, kernel, stride, padding, dilation)
                    .unwrap(),
            );
        }
        out
    }

    /// `n` values uniform in [-1, 1) from `Rng::seed(seed)`.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::seed(seed);
        (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
    }

    /// im2col then a plain product by the weight gives the direct loops' output, bias added,
    /// to within rounding (the sums run in another order), in f64.
    #[test]
    fn im2col_product_matches_direct_loops() {
        for (k, g) in geometries().iter().enumerate() {
            let batch = 2;
            let x = values(batch * g.in_width(), k as u64);
            let wt = values(g.weight_len(), 100 + k as u64);
            let b = values(g.out_channels(), 200 + k as u64);
            let mut direct = vec![0.0; batch * g.out_width()];
            conv2d_reference(g, batch, &x, &wt, Some(&b), &mut direct);
            let (oh, ow) = g.out_size();
            let n_cols = batch * oh * ow;
            let mut cols = vec![f64::NAN; g.kernel_len() * n_cols];
            im2col(g, batch, &x, &mut cols);
            for o in 0..g.out_channels() {
                for col in 0..n_cols {
                    let mut acc = b[o];
                    for r in 0..g.kernel_len() {
                        acc += wt[o * g.kernel_len() + r] * cols[r * n_cols + col];
                    }
                    let (n, p) = (col / (oh * ow), col % (oh * ow));
                    let want = direct[(n * g.out_channels() + o) * oh * ow + p];
                    assert!(
                        (acc - want).abs() < 1e-12,
                        "geometry {k}: {acc} against {want}"
                    );
                }
            }
        }
    }

    /// Geometries the direct kernels run at every level: widths of 8, 16 and 24, one row,
    /// 3 to 12 input channels, 4 to 12 output channels.
    fn direct_geometries() -> Vec<Conv2dGeom> {
        [
            (4, 8, (5, 8)),
            (3, 4, (7, 16)),
            (8, 4, (1, 8)),
            (5, 12, (6, 24)),
            (12, 8, (9, 16)),
        ]
        .into_iter()
        .map(|(cin, cout, input)| {
            Conv2dGeom::with_options(cin, cout, input, (3, 3), (1, 1), (1, 1), (1, 1)).unwrap()
        })
        .collect()
    }

    /// The f32 values of `values(n, seed)`.
    fn values32(n: usize, seed: u64) -> Vec<f32> {
        values(n, seed).into_iter().map(|v| v as f32).collect()
    }

    /// The bits of `v`.
    fn bits32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// `weight` (`[out, in, 3, 3]`) transposed and turned by 180°: the weight of the input
    /// gradient's convolution.
    fn flipped(g: &Conv2dGeom, weight: &[f32]) -> Vec<f32> {
        let (cin, cout) = (g.in_channels(), g.out_channels());
        let mut out = vec![0.0; weight.len()];
        for o in 0..cout {
            for c in 0..cin {
                for k in 0..9 {
                    out[(c * cout + o) * 9 + 8 - k] = weight[(o * cin + c) * 9 + k];
                }
            }
        }
        out
    }

    /// Each direct kernel gives its scalar twin's bits, forward with and without bias and for
    /// the input gradient written and added, on 1 and 3 threads; and the result does not
    /// depend on the thread count.
    #[test]
    fn direct_kernels_match_their_twin() {
        let levels: Vec<Level> = Level::ALL
            .iter()
            .copied()
            .filter(|&l| crate::kernels::supported(l))
            .collect();
        for level in levels {
            for (k, g) in direct_geometries().iter().enumerate() {
                if !direct_runs(level, g) {
                    continue;
                }
                let batch = 3;
                let x = values32(batch * g.in_width(), 1000 + k as u64);
                let w = values32(g.weight_len(), 1100 + k as u64);
                let b = values32(g.out_channels(), 1200 + k as u64);
                let mut scratch = vec![f32::NAN; direct_scratch(g, 3)];
                for bias in [None, Some(&b[..])] {
                    let mut twin = vec![0.0; batch * g.out_width()];
                    conv3x3_direct_twin(level, g, batch, &x, &w, bias, &mut twin, false);
                    for threads in [1, 3] {
                        let mut fast = vec![f32::NAN; batch * g.out_width()];
                        conv3x3_direct(
                            level,
                            g,
                            batch,
                            &x,
                            &w,
                            bias,
                            &mut fast,
                            &mut scratch,
                            threads,
                        );
                        assert_eq!(
                            bits32(&fast),
                            bits32(&twin),
                            "{level:?} geometry {k} forward, {threads} threads"
                        );
                    }
                }
                if !direct_input_grad_runs(level, g) {
                    continue;
                }
                let dy = values32(batch * g.out_width(), 1300 + k as u64);
                let back = Conv2dGeom::with_options(
                    g.out_channels(),
                    g.in_channels(),
                    g.input(),
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                )
                .unwrap();
                let start = values32(batch * g.in_width(), 1400 + k as u64);
                for add in [false, true] {
                    let mut twin = start.clone();
                    conv3x3_direct_twin(
                        level,
                        &back,
                        batch,
                        &dy,
                        &flipped(g, &w),
                        None,
                        &mut twin,
                        add,
                    );
                    for threads in [1, 3] {
                        let mut fast = start.clone();
                        conv3x3_direct_input_grad(
                            level,
                            g,
                            batch,
                            &dy,
                            &w,
                            &mut fast,
                            add,
                            &mut scratch,
                            threads,
                        );
                        assert_eq!(
                            bits32(&fast),
                            bits32(&twin),
                            "{level:?} geometry {k} input gradient, add {add}, {threads} threads"
                        );
                    }
                }
            }
        }
    }

    /// Every weight-gradient kernel this CPU runs gives its twin's bits, written and added,
    /// at 1, 2 and 3 threads, on the direct geometries and on ones with output channels that
    /// are not a multiple of 4 or of the eight-channel block.
    #[test]
    fn direct_weight_grad_matches_its_twin() {
        let levels: Vec<Level> = Level::ALL
            .iter()
            .copied()
            .filter(|&l| crate::kernels::supported(l))
            .collect();
        let mut geometries = direct_geometries();
        for (cin, cout, input) in [(1, 1, (2, 8)), (3, 9, (4, 8)), (2, 19, (3, 16))] {
            geometries.push(
                Conv2dGeom::with_options(cin, cout, input, (3, 3), (1, 1), (1, 1), (1, 1)).unwrap(),
            );
        }
        for level in levels {
            for (k, g) in geometries.iter().enumerate() {
                if !direct_weight_grad_runs(level, g) {
                    continue;
                }
                let batch = 3;
                let x = values32(batch * g.in_width(), 2000 + k as u64);
                let dy = values32(batch * g.out_width(), 2100 + k as u64);
                let start = values32(g.weight_len(), 2200 + k as u64);
                let mut scratch = vec![f32::NAN; direct_weight_grad_scratch(g, 3)];
                for add in [false, true] {
                    let mut twin = start.clone();
                    conv3x3_direct_weight_grad_twin(level, g, batch, &x, &dy, &mut twin, add);
                    for threads in [1, 2, 3] {
                        let mut fast = start.clone();
                        conv3x3_direct_weight_grad(
                            level,
                            g,
                            batch,
                            &x,
                            &dy,
                            &mut fast,
                            add,
                            &mut scratch,
                            threads,
                        );
                        assert_eq!(
                            bits32(&fast),
                            bits32(&twin),
                            "{level:?} geometry {k} weight gradient, add {add}, {threads} threads"
                        );
                    }
                }
            }
        }
    }

    /// The weight-gradient twin agrees with the f64 direct loops within the bound of its
    /// longest chain of rounded steps: at most `2·h·w` within an image (a multiply and an
    /// add per position, at SSE4.2), 3 for the lanes' tree and one per image, so
    /// `|error| <= (2·h·w + batch + 4) · 2⁻²⁴ · Σ|dy·x|`, plus `2⁻¹²⁶` for subnormal
    /// flushing, at both rounding forms and both lane counts.
    #[test]
    fn direct_weight_grad_twin_within_the_sum_bound() {
        for level in [Level::Sse42, Level::Avx2Fma] {
            for (k, g) in direct_geometries().iter().enumerate() {
                if !g
                    .input()
                    .1
                    .is_multiple_of(if level == Level::Sse42 { 4 } else { 8 })
                {
                    continue;
                }
                let batch = 2;
                let x = values32(batch * g.in_width(), 2300 + k as u64);
                let w = values32(g.weight_len(), 2400 + k as u64);
                let dy = values32(batch * g.out_width(), 2500 + k as u64);
                let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
                let abs = |v: &[f64]| v.iter().map(|a| a.abs()).collect::<Vec<f64>>();
                let (x64, w64, dy64) = (wide(&x), wide(&w), wide(&dy));
                let mut dx = vec![0.0; batch * g.in_width()];
                let mut want = vec![0.0; g.weight_len()];
                let mut db = vec![0.0; g.out_channels()];
                conv2d_backward_reference(g, batch, &x64, &w64, &dy64, &mut dx, &mut want, &mut db);
                let mut mass = vec![0.0; g.weight_len()];
                conv2d_backward_reference(
                    g,
                    batch,
                    &abs(&x64),
                    &w64,
                    &abs(&dy64),
                    &mut dx,
                    &mut mass,
                    &mut db,
                );
                let mut got = vec![0.0; g.weight_len()];
                conv3x3_direct_weight_grad_twin(level, g, batch, &x, &dy, &mut got, false);
                let (h, wd) = g.input();
                let gamma = (2 * h * wd + batch + 4) as f64 * 2f64.powi(-24);
                for i in 0..got.len() {
                    let err = (f64::from(got[i]) - want[i]).abs();
                    assert!(
                        err <= gamma * mass[i] + 2f64.powi(-126),
                        "{level:?} geometry {k} weight gradient {i}: {err} > {}",
                        gamma * mass[i]
                    );
                }
            }
        }
    }

    /// The twin's forward pass and input gradient (through the flipped weight) agree with the
    /// f64 direct loops within the bound of a sum of `9·cin + 1` rounded steps:
    /// `|error| <= (9·cin + 2) · 2⁻²⁴ · Σ|x·w|`, plus `2⁻¹²⁶` for subnormal flushing, at
    /// both rounding forms.
    #[test]
    fn direct_twin_within_the_sum_bound() {
        for level in [Level::Sse42, Level::Avx2Fma] {
            for (k, g) in direct_geometries().iter().enumerate() {
                let batch = 2;
                let x = values32(batch * g.in_width(), 1500 + k as u64);
                let w = values32(g.weight_len(), 1600 + k as u64);
                let b = values32(g.out_channels(), 1700 + k as u64);
                let wide = |v: &[f32]| v.iter().map(|&a| f64::from(a)).collect::<Vec<f64>>();
                let abs = |v: &[f64]| v.iter().map(|a| a.abs()).collect::<Vec<f64>>();
                let (x64, w64, b64) = (wide(&x), wide(&w), wide(&b));
                let mut want = vec![0.0; batch * g.out_width()];
                conv2d_reference(g, batch, &x64, &w64, Some(&b64), &mut want);
                let mut mass = vec![0.0; batch * g.out_width()];
                conv2d_reference(
                    g,
                    batch,
                    &abs(&x64),
                    &abs(&w64),
                    Some(&abs(&b64)),
                    &mut mass,
                );
                let mut got = vec![0.0; batch * g.out_width()];
                conv3x3_direct_twin(level, g, batch, &x, &w, Some(&b), &mut got, false);
                let gamma = (9 * g.in_channels() + 2) as f64 * 2f64.powi(-24);
                for i in 0..got.len() {
                    let err = (f64::from(got[i]) - want[i]).abs();
                    assert!(
                        err <= gamma * mass[i] + 2f64.powi(-126),
                        "{level:?} geometry {k} forward {i}: {err} > {}",
                        gamma * mass[i]
                    );
                }
                // Input gradient: the backward loops' dx against the twin on the flipped
                // weight.
                let dy = values32(batch * g.out_width(), 1800 + k as u64);
                let dy64 = wide(&dy);
                let mut dx = vec![0.0; batch * g.in_width()];
                let mut dw = vec![0.0; g.weight_len()];
                let mut db = vec![0.0; g.out_channels()];
                conv2d_backward_reference(g, batch, &x64, &w64, &dy64, &mut dx, &mut dw, &mut db);
                let mut dmass = vec![0.0; batch * g.in_width()];
                conv2d_backward_reference(
                    g,
                    batch,
                    &x64,
                    &abs(&w64),
                    &abs(&dy64),
                    &mut dmass,
                    &mut dw,
                    &mut db,
                );
                let back = Conv2dGeom::with_options(
                    g.out_channels(),
                    g.in_channels(),
                    g.input(),
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                )
                .unwrap();
                let mut gdx = vec![0.0; batch * g.in_width()];
                conv3x3_direct_twin(
                    level,
                    &back,
                    batch,
                    &dy,
                    &flipped(g, &w),
                    None,
                    &mut gdx,
                    false,
                );
                let gamma = (9 * g.out_channels() + 2) as f64 * 2f64.powi(-24);
                for i in 0..gdx.len() {
                    let err = (f64::from(gdx[i]) - dx[i]).abs();
                    assert!(
                        err <= gamma * dmass[i] + 2f64.powi(-126),
                        "{level:?} geometry {k} input gradient {i}: {err}"
                    );
                }
            }
        }
    }

    /// col2im is the transpose of im2col: `<im2col(x), c> = <x, col2im(c)>` for random `x`
    /// and `c`.
    #[test]
    fn col2im_is_the_adjoint_of_im2col() {
        for (k, g) in geometries().iter().enumerate() {
            let batch = 3;
            let (oh, ow) = g.out_size();
            let x = values(batch * g.in_width(), 300 + k as u64);
            let c = values(g.kernel_len() * batch * oh * ow, 400 + k as u64);
            let mut cols = vec![0.0; c.len()];
            im2col(g, batch, &x, &mut cols);
            let mut back = vec![0.0; x.len()];
            col2im(g, batch, &c, &mut back);
            let lhs: f64 = cols.iter().zip(&c).map(|(a, b)| a * b).sum();
            let rhs: f64 = x.iter().zip(&back).map(|(a, b)| a * b).sum();
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "geometry {k}: {lhs} against {rhs}"
            );
        }
    }

    /// The direct backward loops agree with central differences of the direct forward loops
    /// on every geometry, in f64.
    #[test]
    fn backward_reference_matches_differences() {
        for (k, g) in geometries().iter().enumerate() {
            let batch = 2;
            let x = values(batch * g.in_width(), 500 + k as u64);
            let wt = values(g.weight_len(), 600 + k as u64);
            let b = values(g.out_channels(), 700 + k as u64);
            let up = values(batch * g.out_width(), 800 + k as u64);
            let loss = |x: &[f64], wt: &[f64], b: &[f64]| {
                let mut out = vec![0.0; up.len()];
                conv2d_reference(g, batch, x, wt, Some(b), &mut out);
                out.iter().zip(&up).map(|(o, u)| o * u).sum::<f64>()
            };
            let (mut dx, mut dw, mut db) =
                (vec![0.0; x.len()], vec![0.0; wt.len()], vec![0.0; b.len()]);
            conv2d_backward_reference(g, batch, &x, &wt, &up, &mut dx, &mut dw, &mut db);
            let h = 1e-6;
            let check = |grad: &[f64], which: usize| {
                for (i, &an) in grad.iter().enumerate() {
                    let (mut xp, mut wp, mut bp) = (x.clone(), wt.clone(), b.clone());
                    let (mut xm, mut wm, mut bm) = (x.clone(), wt.clone(), b.clone());
                    match which {
                        0 => (xp[i] += h, xm[i] -= h),
                        1 => (wp[i] += h, wm[i] -= h),
                        _ => (bp[i] += h, bm[i] -= h),
                    };
                    let num = (loss(&xp, &wp, &bp) - loss(&xm, &wm, &bm)) / (2.0 * h);
                    assert!(
                        (num - an).abs() <= 1e-7 + 1e-6 * num.abs(),
                        "geometry {k}, gradient {which}[{i}]: {an} against {num}"
                    );
                }
            };
            check(&dx, 0);
            check(&dw, 1);
            check(&db, 2);
        }
    }
}
