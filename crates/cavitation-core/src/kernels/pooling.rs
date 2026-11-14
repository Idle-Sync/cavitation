//! 2-D pooling: max, average and adaptive average pooling over images stored as flat NCHW
//! rows, as PyTorch's `nn.MaxPool2d`, `nn.AvgPool2d` and `nn.AdaptiveAvgPool2d` compute
//! them.
//!
//! Pooling is cheap next to the convolutions around it, so these plain loops are both the
//! implementation and the reference; every window is read in row-major order.
//!
//! # Examples
//!
//! ```
//! use cavitation_core::kernels::pooling::{Pool2dGeom, max_pool2d};
//! // One 4 x 4 image, 2 x 2 windows with stride 2.
//! let geom = Pool2dGeom::new(1, (4, 4), (2, 2)).unwrap();
//! let x: Vec<f32> = (0..16).map(|v| v as f32).collect();
//! let mut out = vec![0.0; 4];
//! let mut winners = vec![0; 4];
//! max_pool2d(&geom, 1, &x, &mut out, Some(&mut winners));
//! assert_eq!(out, [5.0, 7.0, 13.0, 15.0]);
//! ```

use crate::{Error, Real};

/// The shape of a 2-D pooling: channels and image size, window, stride, zero padding and
/// dilation (each per axis, height first), and whether the output size rounds up
/// (`ceil_mode`), checked when it is made.
///
/// The output size follows PyTorch: along each axis,
/// `(size + 2 · padding − dilation · (kernel − 1) − 1) / stride + 1`, rounded down, or up
/// with `ceil_mode`; with `ceil_mode`, a last window that would start in the padding past
/// the image is dropped. The padding may be at most half the window, as PyTorch requires,
/// and a geometry in which some window's taps all miss the image (possible with dilation)
/// is refused, so every window holds at least one pixel of the image.
///
/// ```
/// use cavitation_core::kernels::pooling::Pool2dGeom;
/// let geom = Pool2dGeom::new(16, (32, 32), (2, 2)).unwrap();
/// assert_eq!(geom.out_size(), (16, 16));
/// assert_eq!(geom.out_width(), 16 * 16 * 16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pool2dGeom {
    /// Channels of each image; pooling keeps them.
    channels: usize,
    /// Height and width of each input image.
    input: (usize, usize),
    /// Window height and width.
    kernel: (usize, usize),
    /// Step between windows, down and across.
    stride: (usize, usize),
    /// Zeros (or, for max pooling, positions that never win) added on each side.
    padding: (usize, usize),
    /// Spacing between the window's taps (max pooling only; 1 for average pooling).
    dilation: (usize, usize),
    /// Whether the output size rounds up.
    ceil_mode: bool,
}

/// The output size along one axis, or `None` if no window fits.
fn axis_out(
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil: bool,
) -> Option<usize> {
    let span = dilation.checked_mul(kernel - 1)?.checked_add(1)?;
    let padded = size.checked_add(padding.checked_mul(2)?)?;
    let room = padded.checked_sub(span)?;
    let mut out = if ceil {
        room.div_ceil(stride) + 1
    } else {
        room / stride + 1
    };
    // PyTorch: with ceil_mode, the last window must start inside the image or the left
    // padding.
    if ceil && (out - 1) * stride >= size + padding {
        out -= 1;
    }
    Some(out)
}

impl Pool2dGeom {
    /// A pooling of `channels` images of `input` (height, width) with windows of `kernel`
    /// and stride equal to the window, no padding, no dilation and `ceil_mode` off: PyTorch's
    /// defaults.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if any count or size is 0 or the window is larger than the
    /// image, naming the sizes.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(6, (28, 28), (2, 2)).unwrap().out_size(), (14, 14));
    /// assert!(Pool2dGeom::new(6, (1, 1), (2, 2)).is_err());
    /// ```
    pub fn new(
        channels: usize,
        input: (usize, usize),
        kernel: (usize, usize),
    ) -> Result<Pool2dGeom, Error> {
        Pool2dGeom::with_options(channels, input, kernel, kernel, (0, 0), (1, 1), false)
    }

    /// A pooling with every setting given: stride, padding and dilation (each down, then
    /// across) and `ceil_mode`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidData`] if any count, size, stride or dilation is 0, the padding is more
    /// than half the dilated window on either axis, or no window fits, naming the sizes.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// // 3 x 3 windows with stride 2 and padding 1, rounding up: 7 -> 4.
    /// let geom = Pool2dGeom::with_options(1, (7, 7), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap();
    /// assert_eq!(geom.out_size(), (4, 4));
    /// assert!(Pool2dGeom::with_options(1, (7, 7), (2, 2), (2, 2), (2, 2), (1, 1), false).is_err());
    /// ```
    pub fn with_options(
        channels: usize,
        input: (usize, usize),
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Result<Pool2dGeom, Error> {
        let fail = |what: String| Err(Error::InvalidData(format!("pool2d: {what}")));
        if channels == 0 || input.0 == 0 || input.1 == 0 || kernel.0 == 0 || kernel.1 == 0 {
            return fail(format!(
                "the channels, image and window must be at least 1, got {channels} channels, \
                 image {input:?} and window {kernel:?}"
            ));
        }
        if stride.0 == 0 || stride.1 == 0 || dilation.0 == 0 || dilation.1 == 0 {
            return fail(format!(
                "the stride and dilation must be at least 1, got stride {stride:?} and \
                 dilation {dilation:?}"
            ));
        }
        if padding.0 > kernel.0 / 2 || padding.1 > kernel.1 / 2 {
            return fail(format!(
                "the padding {padding:?} must be at most half the window {kernel:?}"
            ));
        }
        let geom = Pool2dGeom {
            channels,
            input,
            kernel,
            stride,
            padding,
            dilation,
            ceil_mode,
        };
        let rows = axis_out(
            input.0, kernel.0, stride.0, padding.0, dilation.0, ceil_mode,
        );
        let cols = axis_out(
            input.1, kernel.1, stride.1, padding.1, dilation.1, ceil_mode,
        );
        let (Some(rows), Some(cols)) = (rows, cols) else {
            return fail(format!(
                "a {kernel:?} window with dilation {dilation:?} does not fit an image of \
                 {input:?} with padding {padding:?}"
            ));
        };
        // With dilation a window's taps can all miss the image (falling in the padding on
        // both sides); PyTorch writes an index outside the image there, which is refused here.
        let misses = |out: usize, k: usize, st: usize, pd: usize, dl: usize, size: usize| {
            (0..out).find(|&o| {
                (0..k).all(|t| (o * st + t * dl).checked_sub(pd).is_none_or(|v| v >= size))
            })
        };
        let missing = misses(rows, kernel.0, stride.0, padding.0, dilation.0, input.0)
            .map(|o| ("row", o))
            .or(
                misses(cols, kernel.1, stride.1, padding.1, dilation.1, input.1)
                    .map(|o| ("column", o)),
            );
        if let Some((axis, o)) = missing {
            return fail(format!(
                "output {axis} {o}'s window holds no pixel of the image: a {kernel:?} window \
                 with dilation {dilation:?}, stride {stride:?} and padding {padding:?} on an \
                 image of {input:?}"
            ));
        }
        let sizes = [
            channels
                .checked_mul(input.0)
                .and_then(|v| v.checked_mul(input.1)),
            channels.checked_mul(rows).and_then(|v| v.checked_mul(cols)),
        ];
        if sizes.iter().any(Option::is_none) {
            return fail(format!("the sizes overflow: {geom:?}"));
        }
        Ok(geom)
    }

    /// Channels of each image.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 4), (2, 2)).unwrap().channels(), 3);
    /// ```
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Height and width of each input image.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 2)).unwrap().input(), (4, 6));
    /// ```
    pub fn input(&self) -> (usize, usize) {
        self.input
    }

    /// Window height and width.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 3)).unwrap().kernel(), (2, 3));
    /// ```
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    /// The stride, down and across.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 3)).unwrap().stride(), (2, 3));
    /// ```
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// The padding on each side, top and bottom, left and right.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 3)).unwrap().padding(), (0, 0));
    /// ```
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// The dilation, down and across.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 3)).unwrap().dilation(), (1, 1));
    /// ```
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// Whether the output size rounds up.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert!(!Pool2dGeom::new(3, (4, 6), (2, 3)).unwrap().ceil_mode());
    /// ```
    pub fn ceil_mode(&self) -> bool {
        self.ceil_mode
    }

    /// Height and width of each output image.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(1, (5, 7), (2, 2)).unwrap().out_size(), (2, 3));
    /// ```
    pub fn out_size(&self) -> (usize, usize) {
        let along = |a: usize| {
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
            axis_out(size, kernel, stride, padding, dilation, self.ceil_mode)
                .expect("checked when made")
        };
        (along(0), along(1))
    }

    /// Values in one input image: `channels · height · width`.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 2)).unwrap().in_width(), 72);
    /// ```
    pub fn in_width(&self) -> usize {
        self.channels * self.input.0 * self.input.1
    }

    /// Values in one output image: `channels · out_height · out_width`.
    ///
    /// ```
    /// use cavitation_core::kernels::pooling::Pool2dGeom;
    /// assert_eq!(Pool2dGeom::new(3, (4, 6), (2, 2)).unwrap().out_width(), 3 * 2 * 3);
    /// ```
    pub fn out_width(&self) -> usize {
        let (oh, ow) = self.out_size();
        self.channels * oh * ow
    }

    /// The geometry as [`WORDS`] numbers, in the order [`from_words`](Self::from_words) reads.
    pub(crate) fn to_words(self) -> [usize; WORDS] {
        [
            self.channels,
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
            usize::from(self.ceil_mode),
        ]
    }

    /// The geometry [`to_words`](Self::to_words) wrote.
    pub(crate) fn from_words(w: &[usize]) -> Pool2dGeom {
        Pool2dGeom {
            channels: w[0],
            input: (w[1], w[2]),
            kernel: (w[3], w[4]),
            stride: (w[5], w[6]),
            padding: (w[7], w[8]),
            dilation: (w[9], w[10]),
            ceil_mode: w[11] == 1,
        }
    }
}

/// How many numbers [`Pool2dGeom::to_words`] writes.
pub(crate) const WORDS: usize = 12;

/// Panics, naming `what`, unless `len` is `expected`.
fn check_len(what: &str, len: usize, expected: usize) {
    assert!(
        len == expected,
        "pool2d: {what} holds {len} values, expected {expected}"
    );
}

/// Max pooling of `batch` images `x` into `out`, as PyTorch's CPU kernel computes it: each
/// output is the largest value of its window, the taps read in row-major order with a later
/// one winning when `value > best` or `value` is NaN (so a NaN wins, and the last NaN of a
/// window wins over an earlier one), padding never winning. `winners`, if given, receives
/// for each output the position of its winner within its input plane (`row · width +
/// column`), which the backward pass sends the gradient to.
///
/// # Panics
///
/// Panics if a slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::pooling::{Pool2dGeom, max_pool2d};
/// let geom = Pool2dGeom::new(1, (1, 4), (1, 2)).unwrap();
/// let (mut out, mut winners) = (vec![0.0f32; 2], vec![0; 2]);
/// max_pool2d(&geom, 1, &[3.0, 3.0, f32::NAN, 1.0], &mut out, Some(&mut winners));
/// assert_eq!(out[0], 3.0);
/// assert!(out[1].is_nan());
/// assert_eq!(winners, [0, 2]); // a tie goes to the first; NaN wins
/// ```
pub fn max_pool2d<T: Real>(
    geom: &Pool2dGeom,
    batch: usize,
    x: &[T],
    out: &mut [T],
    mut winners: Option<&mut [usize]>,
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the output", out.len(), batch * geom.out_width());
    if let Some(winners) = winners.as_deref() {
        check_len("the winners", winners.len(), batch * geom.out_width());
    }
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw), (dh, dw)) = (geom.stride, geom.padding, geom.dilation);
    for (p, plane) in x.chunks_exact(h * w).enumerate() {
        let o_plane = &mut out[p * oh * ow..][..oh * ow];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = T::from_f64(f64::NEG_INFINITY);
                let mut at = usize::MAX;
                for i in 0..kh {
                    let Some(y) = (oy * sh + i * dh).checked_sub(ph).filter(|&y| y < h) else {
                        continue;
                    };
                    for j in 0..kw {
                        let Some(xx) = (ox * sw + j * dw).checked_sub(pw).filter(|&v| v < w) else {
                            continue;
                        };
                        let v = plane[y * w + xx];
                        #[allow(clippy::eq_op)] // `v != v` is the NaN test `Real` offers
                        let nan = v != v;
                        if v > best || nan || at == usize::MAX {
                            best = v;
                            at = y * w + xx;
                        }
                    }
                }
                debug_assert!(at != usize::MAX, "every window holds a pixel of the image");
                o_plane[oy * ow + ox] = best;
                if let Some(winners) = winners.as_deref_mut() {
                    winners[(p * oh + oy) * ow + ox] = at;
                }
            }
        }
    }
}

/// Adds each output gradient of [`max_pool2d`] to its winner's position in `dx`, output by
/// output in order. `dx` is added to, not overwritten.
///
/// # Panics
///
/// Panics if a slice has the wrong length, naming it.
///
/// ```
/// use cavitation_core::kernels::pooling::{Pool2dGeom, max_pool2d_backward};
/// let geom = Pool2dGeom::new(1, (1, 4), (1, 2)).unwrap();
/// let mut dx = vec![0.0f32; 4];
/// max_pool2d_backward(&geom, 1, &[1.0, 2.0], &[1, 2], &mut dx);
/// assert_eq!(dx, [0.0, 1.0, 2.0, 0.0]);
/// ```
pub fn max_pool2d_backward<T: Real>(
    geom: &Pool2dGeom,
    batch: usize,
    dy: &[T],
    winners: &[usize],
    dx: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the winners", winners.len(), batch * geom.out_width());
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    for (p, plane) in dx.chunks_exact_mut(h * w).enumerate() {
        let g = &dy[p * oh * ow..][..oh * ow];
        let at = &winners[p * oh * ow..][..oh * ow];
        for (&gv, &a) in g.iter().zip(at) {
            plane[a] += gv;
        }
    }
}

/// The in-image part of average-pooling window `o` along one axis, and the length PyTorch
/// divides by when it counts the padding: the window `[o · stride − padding, + kernel)`,
/// clipped to the padded image for the count and to the image for the sum.
fn avg_window(
    o: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    size: usize,
) -> (usize, usize, usize) {
    let start = (o * stride) as isize - padding as isize;
    let end = (start + kernel as isize).min((size + padding) as isize);
    let counted = (end - start) as usize;
    let lo = start.max(0) as usize;
    let hi = end.min(size as isize).max(start.max(0)) as usize;
    (lo, hi, counted)
}

/// Average pooling of `batch` images `x` into `out`, as PyTorch's `avg_pool2d` computes it
/// (no dilation; `geom`'s dilation must be 1): each output is the sum of its window's pixels
/// inside the image, in row-major order, divided by `divisor_override` if given, else by the
/// window's size counting the padding (`count_include_pad`) but not any overhang past it
/// under `ceil_mode`, else by the number of pixels summed.
///
/// # Panics
///
/// Panics if a slice has the wrong length (naming it), the geometry is dilated, or
/// `divisor_override` is 0.
///
/// ```
/// use cavitation_core::kernels::pooling::{Pool2dGeom, avg_pool2d};
/// let geom = Pool2dGeom::with_options(1, (1, 3), (1, 2), (1, 2), (0, 1), (1, 1), false).unwrap();
/// let mut out = vec![0.0f32; 2];
/// avg_pool2d(&geom, 1, &[2.0, 4.0, 6.0], true, None, &mut out);
/// assert_eq!(out, [1.0, 5.0]); // (0 + 2) / 2 and (4 + 6) / 2
/// avg_pool2d(&geom, 1, &[2.0, 4.0, 6.0], false, None, &mut out);
/// assert_eq!(out, [2.0, 5.0]); // 2 / 1: the padding is not counted
/// ```
pub fn avg_pool2d<T: Real>(
    geom: &Pool2dGeom,
    batch: usize,
    x: &[T],
    count_include_pad: bool,
    divisor_override: Option<usize>,
    out: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    check_avg(geom, divisor_override);
    check_len("the input", x.len(), batch * geom.in_width());
    check_len("the output", out.len(), batch * geom.out_width());
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw)) = (geom.stride, geom.padding);
    for (p, plane) in x.chunks_exact(h * w).enumerate() {
        let o_plane = &mut out[p * oh * ow..][..oh * ow];
        for oy in 0..oh {
            let (y0, y1, cy) = avg_window(oy, kh, sh, ph, h);
            for ox in 0..ow {
                let (x0, x1, cx) = avg_window(ox, kw, sw, pw, w);
                let mut acc = T::ZERO;
                for y in y0..y1 {
                    for v in &plane[y * w + x0..y * w + x1] {
                        acc += *v;
                    }
                }
                let divisor = divisor(
                    divisor_override,
                    count_include_pad,
                    cy * cx,
                    (y1 - y0) * (x1 - x0),
                );
                o_plane[oy * ow + ox] = acc / T::from_f64(divisor as f64);
            }
        }
    }
}

/// The divisor of one average-pooling output: the override, else the window counting the
/// padding, else the pixels summed.
fn divisor(
    divisor_override: Option<usize>,
    count_include_pad: bool,
    counted: usize,
    summed: usize,
) -> usize {
    match divisor_override {
        Some(d) => d,
        None if count_include_pad => counted,
        None => summed,
    }
}

/// Panics unless `geom` is undilated and `divisor_override` is not 0.
fn check_avg(geom: &Pool2dGeom, divisor_override: Option<usize>) {
    assert!(
        geom.dilation == (1, 1),
        "avg_pool2d: average pooling has no dilation, got {:?}",
        geom.dilation
    );
    assert!(
        divisor_override != Some(0),
        "avg_pool2d: the divisor override must not be 0"
    );
}

/// The gradient of [`avg_pool2d`]: each output gradient divided by its divisor and added to
/// every pixel of its window inside the image, output by output in order. `dx` is added to,
/// not overwritten.
///
/// # Panics
///
/// As [`avg_pool2d`].
///
/// ```
/// use cavitation_core::kernels::pooling::{Pool2dGeom, avg_pool2d_backward};
/// let geom = Pool2dGeom::new(1, (1, 4), (1, 2)).unwrap();
/// let mut dx = vec![0.0f32; 4];
/// avg_pool2d_backward(&geom, 1, &[2.0, 4.0], true, None, &mut dx);
/// assert_eq!(dx, [1.0, 1.0, 2.0, 2.0]);
/// ```
pub fn avg_pool2d_backward<T: Real>(
    geom: &Pool2dGeom,
    batch: usize,
    dy: &[T],
    count_include_pad: bool,
    divisor_override: Option<usize>,
    dx: &mut [T],
) {
    let (oh, ow) = geom.out_size();
    let (h, w) = geom.input;
    check_avg(geom, divisor_override);
    check_len("the output gradient", dy.len(), batch * geom.out_width());
    check_len("the input gradient", dx.len(), batch * geom.in_width());
    let (kh, kw) = geom.kernel;
    let ((sh, sw), (ph, pw)) = (geom.stride, geom.padding);
    for (p, plane) in dx.chunks_exact_mut(h * w).enumerate() {
        let g = &dy[p * oh * ow..][..oh * ow];
        for oy in 0..oh {
            let (y0, y1, cy) = avg_window(oy, kh, sh, ph, h);
            for ox in 0..ow {
                let (x0, x1, cx) = avg_window(ox, kw, sw, pw, w);
                let d = divisor(
                    divisor_override,
                    count_include_pad,
                    cy * cx,
                    (y1 - y0) * (x1 - x0),
                );
                let share = g[oy * ow + ox] / T::from_f64(d as f64);
                for y in y0..y1 {
                    for v in &mut plane[y * w + x0..y * w + x1] {
                        *v += share;
                    }
                }
            }
        }
    }
}

/// The window of adaptive-pooling output `o` of `out` along an axis of `size`: PyTorch's
/// `floor(o · size / out) .. ceil((o + 1) · size / out)`.
fn adaptive_window(o: usize, out: usize, size: usize) -> (usize, usize) {
    (o * size / out, ((o + 1) * size).div_ceil(out))
}

/// Adaptive average pooling of `batch` images of `channels` planes of `input` to `output`
/// (height, width), as PyTorch's `adaptive_avg_pool2d` computes it: output `(i, j)` is the
/// mean of the window `floor(i · h / oh) .. ceil((i + 1) · h / oh)` by the same across,
/// summed in row-major order and divided by the window's size.
///
/// # Panics
///
/// Panics if a slice has the wrong length or a size is 0, naming it.
///
/// ```
/// use cavitation_core::kernels::pooling::adaptive_avg_pool2d;
/// let mut out = vec![0.0f32; 1];
/// // Output (1, 1): the mean of each plane, global average pooling.
/// adaptive_avg_pool2d(1, 1, (2, 2), (1, 1), &[1.0, 2.0, 3.0, 6.0], &mut out);
/// assert_eq!(out, [3.0]);
/// ```
pub fn adaptive_avg_pool2d<T: Real>(
    batch: usize,
    channels: usize,
    input: (usize, usize),
    output: (usize, usize),
    x: &[T],
    out: &mut [T],
) {
    let ((h, w), (oh, ow)) = (input, output);
    check_adaptive(input, output);
    check_len("the input", x.len(), batch * channels * h * w);
    check_len("the output", out.len(), batch * channels * oh * ow);
    for (p, plane) in x.chunks_exact(h * w).enumerate() {
        let o_plane = &mut out[p * oh * ow..][..oh * ow];
        for oy in 0..oh {
            let (y0, y1) = adaptive_window(oy, oh, h);
            for ox in 0..ow {
                let (x0, x1) = adaptive_window(ox, ow, w);
                let mut acc = T::ZERO;
                for y in y0..y1 {
                    for v in &plane[y * w + x0..y * w + x1] {
                        acc += *v;
                    }
                }
                // PyTorch divides by the window's height, then by its width.
                o_plane[oy * ow + ox] =
                    acc / T::from_f64((y1 - y0) as f64) / T::from_f64((x1 - x0) as f64);
            }
        }
    }
}

/// Panics unless every size of `input` and `output` is at least 1.
fn check_adaptive(input: (usize, usize), output: (usize, usize)) {
    assert!(
        input.0 > 0 && input.1 > 0 && output.0 > 0 && output.1 > 0,
        "adaptive_avg_pool2d: the sizes must be at least 1, got input {input:?} and output \
         {output:?}"
    );
}

/// The gradient of [`adaptive_avg_pool2d`]: each output gradient divided by its window's
/// size and added to every pixel of the window, output by output in order. `dx` is added to,
/// not overwritten.
///
/// # Panics
///
/// As [`adaptive_avg_pool2d`].
///
/// ```
/// use cavitation_core::kernels::pooling::adaptive_avg_pool2d_backward;
/// let mut dx = vec![0.0f32; 4];
/// adaptive_avg_pool2d_backward(1, 1, (2, 2), (1, 1), &[4.0], &mut dx);
/// assert_eq!(dx, [1.0; 4]);
/// ```
pub fn adaptive_avg_pool2d_backward<T: Real>(
    batch: usize,
    channels: usize,
    input: (usize, usize),
    output: (usize, usize),
    dy: &[T],
    dx: &mut [T],
) {
    let ((h, w), (oh, ow)) = (input, output);
    check_adaptive(input, output);
    check_len("the output gradient", dy.len(), batch * channels * oh * ow);
    check_len("the input gradient", dx.len(), batch * channels * h * w);
    for (p, plane) in dx.chunks_exact_mut(h * w).enumerate() {
        let g = &dy[p * oh * ow..][..oh * ow];
        for oy in 0..oh {
            let (y0, y1) = adaptive_window(oy, oh, h);
            for ox in 0..ow {
                let (x0, x1) = adaptive_window(ox, ow, w);
                let share =
                    g[oy * ow + ox] / T::from_f64((y1 - y0) as f64) / T::from_f64((x1 - x0) as f64);
                for y in y0..y1 {
                    for v in &mut plane[y * w + x0..y * w + x1] {
                        *v += share;
                    }
                }
            }
        }
    }
}

/// Tests for the pooling geometry and the pooling loops.
#[cfg(test)]
mod tests {
    use super::*;

    /// Output sizes worked out from PyTorch's formula, `ceil_mode`'s dropped last window
    /// included.
    #[test]
    fn out_size_by_hand() {
        let g = |input, kernel, stride, padding, dilation, ceil| {
            Pool2dGeom::with_options(1, input, kernel, stride, padding, dilation, ceil)
                .map(|g| g.out_size())
        };
        assert_eq!(
            g((32, 32), (2, 2), (2, 2), (0, 0), (1, 1), false).unwrap(),
            (16, 16)
        );
        assert_eq!(
            g((7, 7), (2, 2), (2, 2), (0, 0), (1, 1), false).unwrap(),
            (3, 3)
        );
        assert_eq!(
            g((7, 7), (2, 2), (2, 2), (0, 0), (1, 1), true).unwrap(),
            (4, 4)
        );
        assert_eq!(
            g((7, 7), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            (4, 4)
        );
        // 6 with window 3, stride 2, padding 1: rounding up gives 4, and the 4th window
        // starts at 3 · 2 = 6, inside the image and its padding (6 < 6 + 1), so it stays.
        assert_eq!(
            g((6, 6), (3, 3), (2, 2), (1, 1), (1, 1), true).unwrap(),
            (4, 4)
        );
        // 5 with window 2, stride 2, padding 1: ceil gives 4, whose window starts at 6 >= 5 +
        // 1, so it is dropped.
        assert_eq!(
            g((5, 5), (2, 2), (2, 2), (1, 1), (1, 1), true).unwrap(),
            (3, 3)
        );
        assert_eq!(
            g((9, 9), (3, 3), (1, 1), (0, 0), (2, 2), false).unwrap(),
            (5, 5)
        );
        assert_eq!(
            g((5, 8), (2, 3), (1, 2), (1, 1), (1, 1), false).unwrap(),
            (6, 4)
        );
    }

    /// Too much padding, a window that does not fit, and zero sizes are refused.
    #[test]
    fn bad_geometry_is_refused() {
        let e = Pool2dGeom::with_options(1, (8, 8), (2, 2), (2, 2), (2, 2), (1, 1), false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("at most half the window"), "{e}");
        assert!(Pool2dGeom::new(1, (1, 1), (2, 2)).is_err());
        assert!(Pool2dGeom::new(0, (4, 4), (2, 2)).is_err());
        assert!(
            Pool2dGeom::with_options(1, (4, 4), (2, 2), (0, 1), (0, 0), (1, 1), false).is_err()
        );
        assert!(
            Pool2dGeom::with_options(1, (4, 4), (2, 2), (1, 1), (0, 0), (0, 1), false).is_err()
        );
    }

    /// A dilated window whose taps all fall in the padding is refused, and the padding limit
    /// is half the undilated window, as PyTorch checks it.
    #[test]
    fn windows_that_miss_the_image_are_refused() {
        let e = Pool2dGeom::with_options(1, (1, 3), (1, 2), (1, 1), (0, 1), (1, 4), false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("holds no pixel of the image"), "{e}");
        // Kernel 3, dilation 2: padding 2 is more than 3 / 2 = 1.
        assert!(
            Pool2dGeom::with_options(1, (9, 9), (3, 3), (1, 1), (2, 2), (2, 2), false).is_err()
        );
        assert!(Pool2dGeom::with_options(1, (9, 9), (3, 3), (1, 1), (1, 1), (2, 2), false).is_ok());
    }

    /// A window covering the whole image gives a 1 × 1 output, the maximum or the mean;
    /// adaptive pooling to a larger output than its input repeats pixels in overlapping
    /// windows; and average pooling without the padding under `ceil_mode` divides by the
    /// pixels summed.
    #[test]
    fn whole_image_larger_output_and_excluded_padding() {
        let g = Pool2dGeom::new(1, (2, 3), (2, 3)).unwrap();
        let x = [1.0f64, 5.0, 2.0, 4.0, 3.0, 9.0];
        let (mut out, mut at) = (vec![0.0; 1], vec![0; 1]);
        max_pool2d(&g, 1, &x, &mut out, Some(&mut at));
        assert_eq!((out[0], at[0]), (9.0, 5));
        avg_pool2d(&g, 1, &x, true, None, &mut out);
        assert_eq!(out, [4.0]);
        let mut out = vec![0.0; 3];
        adaptive_avg_pool2d(1, 1, (1, 2), (1, 3), &[2.0, 4.0], &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0]);
        let g = Pool2dGeom::with_options(1, (1, 5), (1, 2), (1, 2), (0, 1), (1, 1), true).unwrap();
        assert_eq!(g.out_size(), (1, 3));
        let mut out = vec![0.0; 3];
        avg_pool2d(&g, 1, &[2.0, 4.0, 6.0, 8.0, 10.0], false, None, &mut out);
        assert_eq!(out, [2.0, 5.0, 9.0]);
    }

    /// The words round trip, `ceil_mode` and the across dilation included.
    #[test]
    fn words_round_trip() {
        let g = Pool2dGeom::with_options(3, (9, 8), (3, 2), (2, 1), (1, 1), (1, 2), true).unwrap();
        assert_eq!(Pool2dGeom::from_words(&g.to_words()), g);
    }

    /// Ties go to the first tap, padding never wins even when every pixel is negative, a NaN
    /// wins and the last of two NaNs wins, and infinity beats every finite value.
    #[test]
    fn max_pool_edge_cases() {
        let g = Pool2dGeom::with_options(1, (2, 2), (2, 2), (2, 2), (1, 1), (1, 1), false).unwrap();
        assert_eq!(g.out_size(), (2, 2));
        let (mut out, mut at) = (vec![0.0f32; 4], vec![0; 4]);
        max_pool2d(&g, 1, &[-5.0, -6.0, -7.0, -8.0], &mut out, Some(&mut at));
        assert_eq!(out, [-5.0, -6.0, -7.0, -8.0]);
        assert_eq!(at, [0, 1, 2, 3]);
        let g = Pool2dGeom::new(1, (1, 4), (1, 4)).unwrap();
        let (mut out, mut at) = (vec![0.0f32; 1], vec![0; 1]);
        max_pool2d(&g, 1, &[2.0, 2.0, 1.0, 2.0], &mut out, Some(&mut at));
        assert_eq!((out[0], at[0]), (2.0, 0));
        max_pool2d(
            &g,
            1,
            &[f32::NAN, 9.0, f32::NAN, 1.0],
            &mut out,
            Some(&mut at),
        );
        assert!(out[0].is_nan());
        assert_eq!(at[0], 2);
        max_pool2d(
            &g,
            1,
            &[1.0, f32::INFINITY, 3.0, 1e30],
            &mut out,
            Some(&mut at),
        );
        assert_eq!((out[0], at[0]), (f32::INFINITY, 1));
    }

    /// Under `ceil_mode` with `count_include_pad`, a window that runs past the padded edge
    /// divides by the window clipped to the padded image, not the whole window.
    #[test]
    fn ceil_mode_divisor_counts_padding_not_overhang() {
        // 6 wide, window 3, stride 2, no padding, ceil: windows [0, 3), [2, 5) and [4, 7),
        // the last clipped to [4, 6), so its divisor is 2, not 3.
        let g = Pool2dGeom::with_options(1, (1, 6), (1, 3), (1, 2), (0, 0), (1, 1), true).unwrap();
        assert_eq!(g.out_size(), (1, 3));
        let mut out = vec![0.0f32; 3];
        avg_pool2d(&g, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0], true, None, &mut out);
        assert_eq!(out, [2.0, 4.0, 6.0]);
        avg_pool2d(
            &g,
            1,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
            true,
            Some(4),
            &mut out,
        );
        assert_eq!(out, [1.5, 3.0, 3.0]);
    }

    /// Adaptive pooling to the input's own size is the identity, and 7 → 3 uses PyTorch's
    /// overlapping windows [0, 3), [2, 5), [4, 7).
    #[test]
    fn adaptive_windows() {
        let x: Vec<f64> = (0..7).map(f64::from).collect();
        let mut out = vec![0.0; 7];
        adaptive_avg_pool2d(1, 1, (1, 7), (1, 7), &x, &mut out);
        assert_eq!(out, x);
        let mut out = vec![0.0; 3];
        adaptive_avg_pool2d(1, 1, (1, 7), (1, 3), &x, &mut out);
        assert_eq!(out, [1.0, 3.0, 5.0]);
    }

    /// Each backward loop is the transpose of its forward loop: `<f(x), u> = <x, f'(u)>` for
    /// average and adaptive pooling (linear maps), on stride, padding and ceil cases.
    #[test]
    fn backward_loops_are_adjoints() {
        let mut rng = crate::Rng::seed(5);
        let mut draw = |n: usize| (0..n).map(|_| rng.range(-1.0, 1.0)).collect::<Vec<f64>>();
        for (g, include, over) in [
            (
                Pool2dGeom::with_options(2, (7, 6), (3, 2), (2, 2), (1, 1), (1, 1), true).unwrap(),
                true,
                None,
            ),
            (
                Pool2dGeom::with_options(2, (7, 6), (3, 2), (2, 2), (1, 1), (1, 1), true).unwrap(),
                false,
                None,
            ),
            (
                Pool2dGeom::with_options(1, (5, 5), (2, 2), (1, 1), (1, 0), (1, 1), false).unwrap(),
                true,
                Some(3),
            ),
        ] {
            let x = draw(2 * g.in_width());
            let u = draw(2 * g.out_width());
            let mut y = vec![0.0; u.len()];
            avg_pool2d(&g, 2, &x, include, over, &mut y);
            let mut back = vec![0.0; x.len()];
            avg_pool2d_backward(&g, 2, &u, include, over, &mut back);
            let lhs: f64 = y.iter().zip(&u).map(|(a, b)| a * b).sum();
            let rhs: f64 = x.iter().zip(&back).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() < 1e-12, "{g:?}: {lhs} against {rhs}");
        }
        let x = draw(2 * 3 * 7 * 5);
        let u = draw(2 * 3 * 3 * 2);
        let mut y = vec![0.0; u.len()];
        adaptive_avg_pool2d(2, 3, (7, 5), (3, 2), &x, &mut y);
        let mut back = vec![0.0; x.len()];
        adaptive_avg_pool2d_backward(2, 3, (7, 5), (3, 2), &u, &mut back);
        let lhs: f64 = y.iter().zip(&u).map(|(a, b)| a * b).sum();
        let rhs: f64 = x.iter().zip(&back).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12);
    }
}
