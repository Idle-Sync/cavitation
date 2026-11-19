//! The shape the per-channel kernels work on, and the limits of their fixed-tree sums
//! (`reduce.cu`, a port of `cavitation-gpu`'s `kernels/reduce.rs`).

use crate::error::CudaError;

/// The sums' source: the first pass's fixed tree and the second pass's merge. Kernels that
/// sum their own terms are compiled after it, in the same module.
pub(crate) const SOURCE: &str = include_str!("reduce.cu");

/// Threads per block of every per-channel kernel; the sums' tree halves it, so it must be a
/// power of two, and `reduce.cu`'s `BLOCK` is the same.
pub(crate) const BLOCK: u32 = 256;

/// The values one block of the sums' first pass adds into one partial sum: [`BLOCK`] threads
/// of 16 values each (`reduce.cu`'s `PER_LANE`).
pub(crate) const CHUNK: u64 = BLOCK as u64 * 16;

/// The most values a tensor may hold: the kernels index them with `u32`s, with room for a
/// block's threads past the end.
const MAX_VALUES: u64 = 1 << 31;

/// The most channels: the sums' first pass puts the channels on the grid's second
/// dimension, which the driver holds to 65,535 blocks.
const MAX_CHANNELS: u32 = 65_535;

/// The shape of a tensor the per-channel kernels work on: `outer` blocks of `channels`
/// channels of `inner` values each, `[outer][channels][inner]` in row-major order. An NCHW
/// batch is `new(batch, C, h · w)`.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::ChannelDims;
///
/// let conv = ChannelDims::new(128, 32, 32 * 32); // NCHW: batch, channels, h · w
/// assert_eq!((conv.len(), conv.summed()), (128 * 32 * 1024, 128 * 1024));
/// let linear = ChannelDims::rows(128, 10); // an M × N matrix: one channel per column
/// assert_eq!(linear, ChannelDims::new(128, 10, 1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelDims {
    /// The blocks of channels: the batch, or a matrix's rows.
    pub outer: u32,
    /// The channels: a convolution's channels, or a matrix's columns.
    pub channels: u32,
    /// The values of one channel in one block: an image plane's `h · w`, or 1 for a matrix.
    pub inner: u32,
}

impl ChannelDims {
    /// `outer` blocks of `channels` channels of `inner` values.
    pub fn new(outer: u32, channels: u32, inner: u32) -> ChannelDims {
        ChannelDims {
            outer,
            channels,
            inner,
        }
    }

    /// A row-major `m × n` matrix, each column a channel: `new(m, n, 1)`.
    pub fn rows(m: u32, n: u32) -> ChannelDims {
        ChannelDims::new(m, n, 1)
    }

    /// The values of the whole tensor, `outer · channels · inner`, or `u64::MAX` for a shape
    /// too large to count in a `u64`.
    pub fn len(&self) -> u64 {
        u64::try_from(self.len_wide()).unwrap_or(u64::MAX)
    }

    /// The values of the whole tensor, without overflow.
    fn len_wide(&self) -> u128 {
        u128::from(self.outer) * u128::from(self.channels) * u128::from(self.inner)
    }

    /// Whether the tensor holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values summed into each channel's sum, `outer · inner`: the `L` the sums' bound
    /// counts, and batch norm's `m`.
    pub fn summed(&self) -> u64 {
        u64::from(self.outer) * u64::from(self.inner)
    }

    /// Chunks of [`CHUNK`] values per channel: one partial sum each.
    pub(crate) fn chunks(&self) -> u32 {
        // At most 2³¹ / 4096 for a shape `check` accepts.
        self.summed().div_ceil(CHUNK) as u32
    }

    /// Refuses, naming `what` and the dims, a shape with a dimension of 0, more than 2³¹
    /// values or more than 65,535 channels.
    pub(crate) fn check(&self, what: &str) -> Result<(), CudaError> {
        let refuse = |why: String| Err(CudaError::Invalid(format!("{what}: {why}, got {self:?}")));
        if self.is_empty() {
            return refuse("every dimension must be above 0".into());
        }
        if self.len_wide() > u128::from(MAX_VALUES) {
            return refuse(format!(
                "{} values are more than the {MAX_VALUES} the kernels take",
                self.len_wide()
            ));
        }
        if self.channels > MAX_CHANNELS {
            return refuse(format!(
                "more than {MAX_CHANNELS} channels, the grid's limit"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shapes the kernels take, and the three refusals.
    #[test]
    fn shapes_are_checked() {
        assert!(ChannelDims::new(128, 32, 1024).check("t").is_ok());
        assert_eq!(ChannelDims::new(128, 32, 1024).chunks(), 32);
        assert_eq!(ChannelDims::new(1, 3, 4097).chunks(), 2);
        for (dims, why) in [
            (ChannelDims::new(0, 1, 1), "above 0"),
            (
                ChannelDims::new(1 << 16, 1 << 16, 1),
                "more than the 2147483648",
            ),
            (
                ChannelDims::new(65_536, 65_537, 4_294_901_761),
                "more than the 2147483648",
            ),
            (ChannelDims::rows(2, 70_000), "more than 65535 channels"),
        ] {
            let Err(CudaError::Invalid(text)) = dims.check("t") else {
                panic!("{dims:?} accepted");
            };
            assert!(text.contains(why), "{text}");
        }
        assert_eq!(
            ChannelDims::new(65_536, 65_537, 4_294_901_761).len(),
            u64::MAX
        );
    }
}
