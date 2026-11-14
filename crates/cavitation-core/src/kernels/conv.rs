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

use crate::kernels::{MatMut, MatRef};
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
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let windows = oh * ow;
    let n_cols = batch * windows;
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the column matrix", cols.len(), geom.kernel_len() * n_cols);
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
                    for oy in 0..oh {
                        let line = &mut dst[oy * ow..][..ow];
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
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    let windows = oh * ow;
    let n_cols = batch * windows;
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    check_len("the column matrix", cols.len(), geom.kernel_len() * n_cols);
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dw)) = (geom.stride, geom.padding, geom.dilation);
    for n in 0..batch {
        for c in 0..geom.in_channels {
            let plane = &mut dx[(n * geom.in_channels + c) * h * w..][..h * w];
            for i in 0..kh {
                let rows = span(oh, i, sh, ph, dh, h);
                for j in 0..kw {
                    let across = span(ow, j, sw, pw, dw, w);
                    if across.is_empty() {
                        continue;
                    }
                    let src = &cols[((c * kh + i) * kw + j) * n_cols + n * windows..][..windows];
                    let first = across.start * sw + j * dw - pw;
                    for oy in rows.clone() {
                        let y = oy * sh + i * dh - ph;
                        let line = &src[oy * ow + across.start..][..across.len()];
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

/// How many column-matrix values [`conv2d_forward`] and the tape's backward pass aim to hold
/// at once (8 MB in f32): images are taken in chunks of
/// `max(1, COLUMN_BUDGET / (kernel_len · windows))`, so each GEMM is wide enough to use the
/// packed kernels well. An image whose own columns are more than this is taken alone, so the
/// buffer holds `max(COLUMN_BUDGET, one image's columns)` values. The chunk size depends on
/// the geometry and the batch alone, never on the thread count, so results do too.
pub const COLUMN_BUDGET: usize = 1 << 21;

/// The number of images [`conv2d_forward`] takes per GEMM for `geom`, at most `batch`.
///
/// ```
/// use cavitation_core::kernels::conv::{COLUMN_BUDGET, Conv2dGeom, images_per_chunk};
/// let geom = Conv2dGeom::with_options(32, 32, (32, 32), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();
/// assert_eq!(images_per_chunk(&geom, 128), COLUMN_BUDGET / (288 * 1024));
/// assert_eq!(images_per_chunk(&geom, 3), 3);
/// ```
pub fn images_per_chunk(geom: &Conv2dGeom, batch: usize) -> usize {
    let (oh, ow) = geom.out_size();
    let per_image = geom.kernel_len() * oh * ow;
    (COLUMN_BUDGET / per_image).clamp(1, batch.max(1))
}

/// The scratch [`conv2d_forward`] needs for `batch` images of `geom`: the column matrix's
/// length and the product's, for one chunk of images.
///
/// ```
/// use cavitation_core::kernels::conv::{Conv2dGeom, conv2d_scratch};
/// let geom = Conv2dGeom::new(1, 2, (3, 3), (2, 2)).unwrap();
/// assert_eq!(conv2d_scratch(&geom, 1), (4 * 4, 2 * 4));
/// ```
pub fn conv2d_scratch(geom: &Conv2dGeom, batch: usize) -> (usize, usize) {
    let (oh, ow) = geom.out_size();
    let images = images_per_chunk(geom, batch);
    let windows = images * oh * ow;
    (geom.kernel_len() * windows, geom.out_channels() * windows)
}

/// The convolution of `batch` images `x` by `weight` (`[out, kernel_len]` row-major, the
/// `[out, in, kh, kw]` weight's own layout), plus `bias`, into `out`, with `gemm(a, b, c)`
/// computing `c = a · b`: chunk by chunk of [`images_per_chunk`] images, the chunk's
/// column matrix in `cols` times the weight into `prod` (`[out, images · windows]`), then
/// each value moved to its output image with the bias added. A 1 × 1 kernel with stride 1
/// and no padding multiplies each image itself instead. The tape and `InferenceModel` both
/// run this, so they agree bit for bit at every kernel level whose GEMM gives each value
/// the same bits whatever the product's shape (the library's own kernels); at the Accelerate
/// level, whose `sgemm` chooses its blocking from the shape, a chunk of another size (one
/// image against several) can round differently, within the GEMM bound. `cols` and `prod`
/// hold at least
/// [`conv2d_scratch`]'s lengths.
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
/// let (c, p) = conv2d_scratch(&geom, 1);
/// let (mut cols, mut prod, mut out) = (vec![0.0; c], vec![0.0; p], vec![0.0; 4]);
/// let gemm = |a: MatRef<f64>, b: MatRef<f64>, c: MatMut<f64>| reference::gemm(1.0, a, b, 0.0, c);
/// conv2d_forward(&geom, 1, &x, &[1.0; 4], Some(&[0.5]), &mut out, &mut cols, &mut prod, gemm);
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
    cols: &mut [T],
    prod: &mut [T],
    mut gemm: impl FnMut(MatRef<'_, T>, MatRef<'_, T>, MatMut<'_, T>),
) {
    let (oh, ow) = geom.out_size();
    let (windows, kl, cout) = (oh * ow, geom.kernel_len(), geom.out_channels());
    let (in_w, out_w) = (geom.in_width(), geom.out_width());
    check_len("the input", x.len(), batch * in_w);
    check_len("the weight", weight.len(), geom.weight_len());
    check_len("the output", out.len(), batch * out_w);
    let wm = MatRef::new(weight, cout, kl, kl, 1).expect("checked length");
    if geom.is_pointwise() {
        for (image, y) in x.chunks_exact(in_w).zip(out.chunks_exact_mut(out_w)) {
            let rhs = MatRef::new(image, kl, windows, windows, 1).expect("checked length");
            gemm(
                wm,
                rhs,
                MatMut::new(y, cout, windows, windows, 1).expect("checked length"),
            );
        }
    } else {
        let per = images_per_chunk(geom, batch);
        let mut n0 = 0;
        while n0 < batch {
            let nb = per.min(batch - n0);
            let cols_n = nb * windows;
            let cm = &mut cols[..kl * cols_n];
            im2col(geom, nb, &x[n0 * in_w..][..nb * in_w], cm);
            let pm = &mut prod[..cout * cols_n];
            gemm(
                wm,
                MatRef::new(&*cm, kl, cols_n, cols_n, 1).expect("checked length"),
                MatMut::new(pm, cout, cols_n, cols_n, 1).expect("checked length"),
            );
            for (k, y) in out[n0 * out_w..][..nb * out_w]
                .chunks_exact_mut(out_w)
                .enumerate()
            {
                for (o, plane) in y.chunks_exact_mut(windows).enumerate() {
                    plane.copy_from_slice(&pm[o * cols_n + k * windows..][..windows]);
                }
            }
            n0 += nb;
        }
    }
    if let Some(b) = bias {
        check_len("the bias", b.len(), cout);
        for image in out.chunks_exact_mut(out_w) {
            for (plane, &bo) in image.chunks_exact_mut(windows).zip(b) {
                for v in plane {
                    *v += bo;
                }
            }
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
