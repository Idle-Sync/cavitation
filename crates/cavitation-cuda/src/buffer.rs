//! Memory on the card: allocated, written from the host and read back.

use std::ffi::c_void;

use crate::device::Cuda;
use crate::error::CudaError;
use crate::sys::DevicePtr;

/// Keeps [`Scalar`] to the types this crate implements it for.
mod sealed {
    /// Implemented for the types [`Scalar`](super::Scalar) is.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// A number a [`DeviceBuffer`] is written from and read into: `f32`, `u32` or `i32`, each
/// four bytes with every bit pattern a value.
pub trait Scalar: sealed::Sealed + Copy {}
impl Scalar for f32 {}
impl Scalar for u32 {}
impl Scalar for i32 {}

/// A block of the card's memory, freed when dropped.
///
/// It starts as zeros. It holds bytes, not a type: what is written as `f32` is read as
/// `f32`, in the host's byte order, which is the card's.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::Cuda;
///
/// if let Some(cuda) = Cuda::new() {
///     let mut buffer = cuda.buffer(3 * 4).unwrap();
///     buffer.write(&[1.0f32, -2.0, 3.5]).unwrap();
///     let mut back = [0.0f32; 3];
///     buffer.read(&mut back).unwrap();
///     assert_eq!(back, [1.0, -2.0, 3.5]);
/// }
/// ```
#[derive(Debug)]
pub struct DeviceBuffer<'c> {
    /// The CUDA it was allocated in.
    cuda: &'c Cuda,
    /// Its address on the card.
    pub(crate) pointer: DevicePtr,
    /// Its size as asked for.
    bytes: usize,
}

impl Cuda {
    /// Allocates `bytes` bytes of the card's memory, set to zero.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver cannot allocate it: `CUDA_ERROR_OUT_OF_MEMORY`
    /// when the card has too little free.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let buffer = cuda.buffer(1024).unwrap();
    ///     assert_eq!(buffer.bytes(), 1024);
    /// }
    /// ```
    pub fn buffer(&self, bytes: usize) -> Result<DeviceBuffer<'_>, CudaError> {
        // The driver refuses an allocation of no bytes, so an empty buffer takes four.
        let allocated = bytes.max(4);
        let _bound = self.bind()?;
        let mut pointer: DevicePtr = 0;
        // SAFETY: the pointer is to a live address, written once, and the card's context
        // is current on this thread, with the lock held.
        let code = unsafe { (self.driver.cuMemAlloc_v2)(&mut pointer, allocated) };
        self.check("cuMemAlloc_v2", code)?;
        // SAFETY: the address was just allocated with `allocated` bytes.
        let code = unsafe { (self.driver.cuMemsetD8_v2)(pointer, 0, allocated) };
        if let Err(error) = self.check("cuMemsetD8_v2", code) {
            // Freed here, under the lock already held: a `DeviceBuffer` dropped now would
            // take it again.
            // SAFETY: the address came from `cuMemAlloc_v2` just above and is freed once.
            unsafe { (self.driver.cuMemFree_v2)(pointer) };
            return Err(error);
        }
        Ok(DeviceBuffer {
            cuda: self,
            pointer,
            bytes,
        })
    }
}

impl DeviceBuffer<'_> {
    /// The buffer's size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copies `data` to the start of the buffer, after the work already given to the card.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver reports an error.
    ///
    /// # Panics
    ///
    /// If `data` is larger than the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut buffer = cuda.buffer(16).unwrap();
    ///     buffer.write(&[7u32, 8]).unwrap();
    /// }
    /// ```
    pub fn write<T: Scalar>(&mut self, data: &[T]) -> Result<(), CudaError> {
        let bytes = size_of_val(data);
        assert!(
            bytes <= self.bytes,
            "{bytes} bytes do not fit a buffer of {}",
            self.bytes
        );
        if bytes == 0 {
            return Ok(());
        }
        let _bound = self.cuda.bind()?;
        // SAFETY: `data` is `bytes` readable bytes, and the buffer holds at least as many.
        let code = unsafe {
            (self.cuda.driver.cuMemcpyHtoD_v2)(self.pointer, data.as_ptr().cast::<c_void>(), bytes)
        };
        self.cuda.check("cuMemcpyHtoD_v2", code)
    }

    /// Copies the start of the buffer into `out`, after the work already given to the
    /// card: a kernel launched before this call has finished when the copy is made.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver reports an error, which is also how a kernel that
    /// failed after its launch returned is reported.
    ///
    /// # Panics
    ///
    /// If `out` is larger than the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let buffer = cuda.buffer(8).unwrap();
    ///     let mut zeros = [1i32; 2];
    ///     buffer.read(&mut zeros).unwrap();
    ///     assert_eq!(zeros, [0, 0]);
    /// }
    /// ```
    pub fn read<T: Scalar>(&self, out: &mut [T]) -> Result<(), CudaError> {
        let bytes = size_of_val(out);
        assert!(
            bytes <= self.bytes,
            "{bytes} bytes are more than a buffer of {}",
            self.bytes
        );
        if bytes == 0 {
            return Ok(());
        }
        let _bound = self.cuda.bind()?;
        // SAFETY: `out` is `bytes` writable bytes, the buffer holds at least as many, and
        // every bit pattern is a value of a `Scalar`.
        let code = unsafe {
            (self.cuda.driver.cuMemcpyDtoH_v2)(
                out.as_mut_ptr().cast::<c_void>(),
                self.pointer,
                bytes,
            )
        };
        self.cuda.check("cuMemcpyDtoH_v2", code)
    }

    /// Copies `bytes` bytes of `from`, starting `from_at` bytes into it, into this buffer
    /// starting `at` bytes in, after the work already given to the card. The copy is queued
    /// on the card and returns without waiting for it.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver reports an error.
    ///
    /// # Panics
    ///
    /// If either range runs past its buffer's end, or the two buffers were allocated in two
    /// different [`Cuda`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut from = cuda.buffer(12).unwrap();
    ///     from.write(&[1.0f32, 2.0, 3.0]).unwrap();
    ///     let mut to = cuda.buffer(12).unwrap();
    ///     to.copy_from(4, &from, 0, 8).unwrap();
    ///     let mut back = [0.0f32; 3];
    ///     to.read(&mut back).unwrap();
    ///     assert_eq!(back, [0.0, 1.0, 2.0]);
    /// }
    /// ```
    pub fn copy_from(
        &mut self,
        at: usize,
        from: &DeviceBuffer<'_>,
        from_at: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        assert!(
            std::ptr::eq(self.cuda, from.cuda),
            "a copy between buffers of two different Cuda handles"
        );
        let fits = |start: usize, len: usize| start.checked_add(bytes).is_some_and(|e| e <= len);
        assert!(
            fits(at, self.bytes) && fits(from_at, from.bytes),
            "a copy of {bytes} bytes from byte {from_at} of a buffer of {} to byte {at} of one \
             of {}",
            from.bytes,
            self.bytes
        );
        if bytes == 0 {
            return Ok(());
        }
        let _bound = self.cuda.bind()?;
        // SAFETY: both ranges lie in live allocations of this context (checked above), and
        // they do not overlap: `self`, borrowed mutably, is not `from`.
        let code = unsafe {
            (self.cuda.driver.cuMemcpyDtoD_v2)(
                self.pointer + at as u64,
                from.pointer + from_at as u64,
                bytes,
            )
        };
        self.cuda.check("cuMemcpyDtoD_v2", code)
    }

    /// Sets every byte of the buffer to zero, after the work already given to the card. The
    /// write is queued on the card and returns without waiting for it.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver reports an error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let mut buffer = cuda.buffer(8).unwrap();
    ///     buffer.write(&[1.5f32, -2.0]).unwrap();
    ///     buffer.zero().unwrap();
    ///     let mut back = [1.0f32; 2];
    ///     buffer.read(&mut back).unwrap();
    ///     assert_eq!(back.map(f32::to_bits), [0, 0]);
    /// }
    /// ```
    pub fn zero(&mut self) -> Result<(), CudaError> {
        if self.bytes == 0 {
            return Ok(());
        }
        let _bound = self.cuda.bind()?;
        // SAFETY: the address was allocated with at least `bytes` bytes in this context.
        let code = unsafe { (self.cuda.driver.cuMemsetD8_v2)(self.pointer, 0, self.bytes) };
        self.cuda.check("cuMemsetD8_v2", code)
    }
}

impl DeviceBuffer<'_> {
    /// Sets the first `words` four-byte words to `value`, queued on the card after its earlier
    /// work, without the host waiting for that work as a [`write`](Self::write) does.
    ///
    /// # Panics
    ///
    /// If the buffer holds fewer than `words` words.
    pub(crate) fn fill_words(&mut self, value: u32, words: usize) -> Result<(), CudaError> {
        assert!(
            words.saturating_mul(4) <= self.bytes,
            "{words} words do not fit a buffer of {} bytes",
            self.bytes
        );
        if words == 0 {
            return Ok(());
        }
        let _bound = self.cuda.bind()?;
        // SAFETY: the address was allocated with at least 4 · `words` bytes (checked above)
        // in this context, current with the lock held.
        let code = unsafe { (self.cuda.driver.cuMemsetD32_v2)(self.pointer, value, words) };
        self.cuda.check("cuMemsetD32_v2", code)
    }
}

impl Drop for DeviceBuffer<'_> {
    /// Waits for the card's work, which may still use the buffer, and frees it. Errors are
    /// not reported: there is no one to report them to.
    fn drop(&mut self) {
        // The lock is taken and the context made current first; if that fails, the calls
        // below fail too, and nothing else can be done.
        let _bound = self.cuda.bind_anyway();
        // SAFETY: the address came from `cuMemAlloc_v2` in the card's context, now current
        // on this thread with the lock held, and is freed once.
        unsafe {
            (self.cuda.driver.cuCtxSynchronize)();
            (self.cuda.driver.cuMemFree_v2)(self.pointer);
        }
    }
}
