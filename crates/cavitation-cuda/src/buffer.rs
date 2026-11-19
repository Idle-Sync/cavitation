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
        let mut pointer: DevicePtr = 0;
        // SAFETY: the pointer is to a live address, written once, and this thread's
        // current context is the one opened.
        let code = unsafe { (self.driver.cuMemAlloc_v2)(&mut pointer, allocated) };
        self.check("cuMemAlloc_v2", code)?;
        let buffer = DeviceBuffer {
            cuda: self,
            pointer,
            bytes,
        };
        // SAFETY: the address was just allocated with `allocated` bytes.
        let code = unsafe { (self.driver.cuMemsetD8_v2)(pointer, 0, allocated) };
        self.check("cuMemsetD8_v2", code)?;
        Ok(buffer)
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
}

impl Drop for DeviceBuffer<'_> {
    /// Waits for the card's work, which may still use the buffer, and frees it. Errors are
    /// not reported: there is no one to report them to.
    fn drop(&mut self) {
        // SAFETY: the address came from `cuMemAlloc_v2` in the context that is current on
        // this thread, and is freed once.
        unsafe {
            (self.cuda.driver.cuCtxSynchronize)();
            (self.cuda.driver.cuMemFree_v2)(self.pointer);
        }
    }
}
