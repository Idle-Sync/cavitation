//! Kernels written in CUDA C++, compiled at run time by NVRTC and launched on the card.

use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;

use crate::buffer::DeviceBuffer;
use crate::device::Cuda;
use crate::error::CudaError;
use crate::sys::{CuFunction, CuModule, Nvrtc};

/// A build-time check that the target is little-endian: a kernel's four-byte arguments are
/// passed as the low bytes of eight-byte slots.
const _: () = assert!(cfg!(target_endian = "little"));

/// The most arguments [`Kernel::launch`] passes.
pub const MAX_KERNEL_ARGS: usize = 16;

/// The option every source is compiled with: no fused multiply-add, so `a * b + c` is two
/// roundings, as on the CPU.
const NO_FMAD: &CStr = c"--fmad=false";

/// Kernels compiled from one CUDA C++ source text, unloaded when dropped.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::Cuda;
///
/// const SOURCE: &str = r#"
///     extern "C" __global__ void twice(float *x, unsigned int n) {
///         unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
///         if (i < n) x[i] = 2.0f * x[i];
///     }
/// "#;
///
/// if let Some(cuda) = Cuda::new() {
///     let module = cuda.compile("twice", SOURCE).unwrap();
///     assert!(module.kernel("twice").is_ok());
///     assert!(module.kernel("thrice").is_err());
/// }
/// ```
#[derive(Debug)]
pub struct Module<'c> {
    /// The CUDA it was loaded in.
    cuda: &'c Cuda,
    /// The driver's handle to the loaded module.
    module: CuModule,
}

/// One kernel of a [`Module`], ready to launch.
#[derive(Debug)]
pub struct Kernel<'m> {
    /// The CUDA its module was loaded in.
    cuda: &'m Cuda,
    /// The driver's handle to the kernel.
    function: CuFunction,
    /// The module the handle is valid for.
    module: PhantomData<&'m Module<'m>>,
}

/// One argument of a kernel launch.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a> {
    /// A pointer parameter: the buffer's address on the card. The buffer may come from any
    /// [`Cuda`], since they share the card's primary context.
    Buffer(&'a DeviceBuffer<'a>),
    /// An `unsigned int` parameter.
    U32(u32),
    /// An `int` parameter.
    I32(i32),
    /// A `float` parameter.
    F32(f32),
}

/// An NVRTC program, destroyed when dropped. It lives within one [`Cuda::compile`] call and
/// is dropped there before the lock is given back.
struct Program<'a> {
    /// NVRTC's functions.
    nvrtc: &'a Nvrtc,
    /// The program's handle.
    raw: *mut c_void,
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle came from `nvrtcCreateProgram` and is destroyed once.
        unsafe { (self.nvrtc.nvrtcDestroyProgram)(&mut self.raw) };
    }
}

impl Program<'_> {
    /// One of the program's texts (the PTX or the log), read with its size function and
    /// its copy function, NUL-terminated.
    fn text(
        &self,
        size_of: unsafe extern "C" fn(*mut c_void, *mut usize) -> i32,
        copy: unsafe extern "C" fn(*mut c_void, *mut c_char) -> i32,
    ) -> Option<Vec<u8>> {
        let mut size = 0usize;
        // SAFETY: the program is live, and the pointer is to a live `usize`.
        if unsafe { size_of(self.raw, &mut size) } != 0 || size == 0 {
            return None;
        }
        let mut text = vec![0u8; size];
        // SAFETY: the buffer has the size the program just reported for this text.
        if unsafe { copy(self.raw, text.as_mut_ptr().cast::<c_char>()) } != 0 {
            return None;
        }
        // The size includes the terminating NUL; it is set again in case it was not written.
        text[size - 1] = 0;
        Some(text)
    }
}

impl Cuda {
    /// Compiles a CUDA C++ source text with NVRTC and loads the result on the card. `name`
    /// is what the compiler's messages call the source.
    ///
    /// A kernel is a function declared `extern "C" __global__ void`, so that
    /// [`Module::kernel`] finds it under the name it is written with. The source is
    /// compiled with `--fmad=false`: a multiply followed by an add is never fused into one
    /// rounding, so a kernel's arithmetic is what the same expressions give on the CPU. It
    /// is compiled for the compute capability in [`CudaInfo::compiles_for`](crate::CudaInfo).
    ///
    /// # Errors
    ///
    /// [`CudaError::Compile`] with the compiler's log if the source does not compile or
    /// holds a NUL character, and [`CudaError::Call`] if the driver does not load the
    /// result.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Cuda, CudaError};
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let err = cuda.compile("broken", "this is not CUDA").unwrap_err();
    ///     assert!(matches!(err, CudaError::Compile { .. }));
    /// }
    /// ```
    pub fn compile(&self, name: &str, source: &str) -> Result<Module<'_>, CudaError> {
        let failed = |log: String| CudaError::Compile {
            name: name.to_string(),
            log,
        };
        let c_source =
            CString::new(source).map_err(|_| failed("the source holds a NUL character".into()))?;
        let c_name =
            CString::new(name).map_err(|_| failed("the name holds a NUL character".into()))?;
        let nvrtc = &self.nvrtc;
        // Held across NVRTC's calls as well as the driver's, and declared before the
        // program, which is destroyed before it is given back.
        let _bound = self.bind()?;

        let mut raw = std::ptr::null_mut();
        // SAFETY: the pointer is to a live handle, both strings are NUL-terminated and
        // live across the call, and no headers are passed; the lock is held.
        let code = unsafe {
            (nvrtc.nvrtcCreateProgram)(
                &mut raw,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if code != 0 {
            return Err(self.nvrtc_error("nvrtcCreateProgram", code));
        }
        let program = Program { nvrtc, raw };

        let arch = CString::new(format!("--gpu-architecture=compute_{}", self.arch()))
            .expect("an architecture option holds no NUL character");
        let options = [NO_FMAD.as_ptr(), arch.as_ptr()];
        // SAFETY: the program is live, and `options` holds as many NUL-terminated strings
        // as the count passed.
        let code = unsafe {
            (nvrtc.nvrtcCompileProgram)(program.raw, options.len() as i32, options.as_ptr())
        };
        if code != 0 {
            let log = program
                .text(nvrtc.nvrtcGetProgramLogSize, nvrtc.nvrtcGetProgramLog)
                .map(|log| {
                    String::from_utf8_lossy(&log[..log.len() - 1])
                        .trim()
                        .to_string()
                })
                .filter(|log| !log.is_empty());
            return Err(failed(log.unwrap_or_else(|| {
                self.nvrtc_error("nvrtcCompileProgram", code).to_string()
            })));
        }
        let ptx = program
            .text(nvrtc.nvrtcGetPTXSize, nvrtc.nvrtcGetPTX)
            .ok_or_else(|| failed("the compiler returned no PTX".into()))?;

        let mut module: CuModule = std::ptr::null_mut();
        // SAFETY: the pointer is to a live handle, `ptx` is a NUL-terminated PTX text, and
        // the context is current on this thread with the lock held.
        let code =
            unsafe { (self.driver.cuModuleLoadData)(&mut module, ptx.as_ptr().cast::<c_void>()) };
        self.check("cuModuleLoadData", code)?;
        Ok(Module { cuda: self, module })
    }
}

impl<'c> Module<'c> {
    /// The kernel the source declares under `name`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] with `CUDA_ERROR_NOT_FOUND` if the module has no kernel of that
    /// name: one that is misspelt, or not declared `extern "C"`.
    ///
    /// # Panics
    ///
    /// If `name` holds a NUL character.
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_>, CudaError> {
        let function = self.function(name)?;
        // SAFETY: the handle was just looked up in this module.
        Ok(unsafe { self.kernel_from(function) })
    }

    /// The handle of the kernel the source declares under `name`, for a type that keeps the
    /// module and looks each kernel up once; [`Module::kernel_from`] makes it a [`Kernel`].
    ///
    /// # Errors
    ///
    /// As [`Module::kernel`].
    ///
    /// # Panics
    ///
    /// If `name` holds a NUL character.
    pub(crate) fn function(&self, name: &str) -> Result<CuFunction, CudaError> {
        let c_name = CString::new(name).expect("a kernel name holds no NUL character");
        let _bound = self.cuda.bind()?;
        let mut function: CuFunction = std::ptr::null_mut();
        // SAFETY: the pointer is to a live handle, the module is loaded, and the name is
        // NUL-terminated and lives across the call.
        let code = unsafe {
            (self.cuda.driver.cuModuleGetFunction)(&mut function, self.module, c_name.as_ptr())
        };
        self.cuda.check("cuModuleGetFunction", code)?;
        Ok(function)
    }

    /// The kernel whose handle is `function`.
    ///
    /// # Safety
    ///
    /// `function` must come from this module's [`Module::function`]: a handle from another
    /// module is invalid once that module is unloaded, and launching it is undefined
    /// behaviour.
    pub(crate) unsafe fn kernel_from(&self, function: CuFunction) -> Kernel<'_> {
        Kernel {
            cuda: self.cuda,
            function,
            module: PhantomData,
        }
    }
}

impl Drop for Module<'_> {
    /// Waits for the card's work, which may still run the module's kernels, and unloads
    /// it. Errors are not reported: there is no one to report them to.
    fn drop(&mut self) {
        // The lock is taken and the context made current first; if that fails, the calls
        // below fail too, and nothing else can be done.
        let _bound = self.cuda.bind_anyway();
        // SAFETY: the module came from `cuModuleLoadData` in the card's context, now
        // current on this thread with the lock held, and is unloaded once.
        unsafe {
            (self.cuda.driver.cuCtxSynchronize)();
            (self.cuda.driver.cuModuleUnload)(self.module);
        }
    }
}

impl Kernel<'_> {
    /// Starts the kernel on `grid[0] × grid[1] × grid[2]` blocks of
    /// `block[0] × block[1] × block[2]` threads, with `args` as its parameters in order,
    /// and returns without waiting for it. A [`DeviceBuffer::read`] after it sees its
    /// results, and [`Cuda::synchronize`] waits for it.
    ///
    /// No host memory is allocated.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver refuses the launch, as it does a grid or block
    /// with a zero or oversized dimension. A kernel that fails while running is reported by
    /// a later call.
    ///
    /// # Panics
    ///
    /// If there are more than [`MAX_KERNEL_ARGS`] arguments.
    ///
    /// # Safety
    ///
    /// `args` must be the kernel's parameters: as many as it declares, each of its
    /// parameter's kind (a buffer for a pointer, `U32` for `unsigned int`, `I32` for `int`,
    /// `F32` for `float`). The driver reads one value per declared parameter, so too few or
    /// too narrow an argument is a read out of bounds on the host. The kernel must stay
    /// within the buffers it is given: what it writes beyond one lands in other memory on
    /// the card.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::{Arg, Cuda};
    ///
    /// const SOURCE: &str = r#"
    ///     extern "C" __global__ void twice(float *x, unsigned int n) {
    ///         unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    ///         if (i < n) x[i] = 2.0f * x[i];
    ///     }
    /// "#;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     let module = cuda.compile("twice", SOURCE).unwrap();
    ///     let twice = module.kernel("twice").unwrap();
    ///     let mut x = cuda.buffer(3 * 4).unwrap();
    ///     x.write(&[1.0f32, 2.5, -4.0]).unwrap();
    ///     // SAFETY: the kernel takes a float pointer and an unsigned count, and writes
    ///     // only the first `n` floats of the buffer, which holds three.
    ///     unsafe { twice.launch([1, 1, 1], [64, 1, 1], &[Arg::Buffer(&x), Arg::U32(3)]) }.unwrap();
    ///     let mut out = [0.0f32; 3];
    ///     x.read(&mut out).unwrap();
    ///     assert_eq!(out, [2.0, 5.0, -8.0]);
    /// }
    /// ```
    pub unsafe fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg<'_>],
    ) -> Result<(), CudaError> {
        assert!(
            args.len() <= MAX_KERNEL_ARGS,
            "a kernel launch takes at most {MAX_KERNEL_ARGS} arguments, not {}",
            args.len()
        );
        // Each argument's value sits in the low bytes of an eight-byte slot, and the
        // driver is given one pointer per slot.
        let mut values = [0u64; MAX_KERNEL_ARGS];
        for (slot, arg) in values.iter_mut().zip(args) {
            *slot = match *arg {
                Arg::Buffer(buffer) => buffer.pointer,
                Arg::U32(value) => u64::from(value),
                Arg::I32(value) => u64::from(value.cast_unsigned()),
                Arg::F32(value) => u64::from(value.to_bits()),
            };
        }
        let _bound = self.cuda.bind()?;
        let mut pointers = [std::ptr::null_mut::<c_void>(); MAX_KERNEL_ARGS];
        for (pointer, value) in pointers.iter_mut().zip(values.iter_mut()) {
            *pointer = std::ptr::from_mut(value).cast::<c_void>();
        }
        // SAFETY: the kernel's module is loaded (the kernel borrows it), `pointers` holds
        // a pointer to a live value for each of the kernel's parameters by this function's
        // contract, no extra options are passed, and the default stream is null.
        let code = unsafe {
            (self.cuda.driver.cuLaunchKernel)(
                self.function,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                std::ptr::null_mut(),
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.cuda.check("cuLaunchKernel", code)
    }
}
