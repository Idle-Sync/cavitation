//! The functions this crate calls in NVIDIA's libraries, one table per library, each
//! function looked up by name when its library is loaded.
//!
//! The signatures are NVIDIA's, from `cuda.h`, `cudnn.h`, `cublas_api.h` and `nvrtc.h`.
//! Every function but the version queries returns a status, 0 for success.

use std::ffi::{c_char, c_int, c_uint, c_void};

use crate::library::Library;

/// A device's ordinal handle (`CUdevice`).
pub(crate) type CuDevice = c_int;
/// A context (`CUcontext`).
pub(crate) type CuContext = *mut c_void;
/// A loaded module of kernels (`CUmodule`).
pub(crate) type CuModule = *mut c_void;
/// One kernel in a module (`CUfunction`).
pub(crate) type CuFunction = *mut c_void;
/// A stream of work (`CUstream`); null is the default stream.
pub(crate) type CuStream = *mut c_void;
/// An address in device memory (`CUdeviceptr`).
pub(crate) type DevicePtr = u64;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
pub(crate) const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
pub(crate) const COMPUTE_CAPABILITY_MINOR: c_int = 76;
/// `CUDA_ERROR_NO_DEVICE`: the driver found no CUDA-capable device.
pub(crate) const ERROR_NO_DEVICE: c_int = 100;

/// Declares a table of functions and how to fill it from a loaded library.
macro_rules! api {
    (
        $(#[$meta:meta])*
        $name:ident, $abi:literal {
            $(
                $(#[$fmeta:meta])*
                fn $f:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;
            )*
        }
    ) => {
        $(#[$meta])*
        #[allow(non_snake_case)]
        pub(crate) struct $name {
            $(
                $(#[$fmeta])*
                pub(crate) $f: unsafe extern $abi fn($($arg: $ty),*) -> $ret,
            )*
        }

        impl $name {
            /// Looks up each of the table's functions in `library` by its name.
            ///
            /// # Errors
            ///
            /// The name of the first function the library does not export.
            pub(crate) fn load(library: &Library) -> Result<Self, &'static str> {
                Ok(Self {
                    $(
                        $f: {
                            let name = concat!(stringify!($f), "\0");
                            let name = std::ffi::CStr::from_bytes_with_nul(name.as_bytes())
                                .expect("a function name holds no NUL");
                            // SAFETY: the field's type is the signature NVIDIA's header
                            // declares for the function of this name, with the calling
                            // convention the header gives the library.
                            unsafe { library.function(name) }.ok_or(stringify!($f))?
                        },
                    )*
                })
            }
        }
    };
}

api! {
    /// The CUDA driver API, in the library NVIDIA's display driver installs.
    Driver, "system" {
        /// Initialises the driver API; `flags` must be 0.
        fn cuInit(flags: c_uint) -> c_int;
        /// The newest CUDA version the driver supports, as `1000 * major + 10 * minor`.
        fn cuDriverGetVersion(version: *mut c_int) -> c_int;
        /// How many CUDA-capable devices there are.
        fn cuDeviceGetCount(count: *mut c_int) -> c_int;
        /// The handle of the device at `ordinal`.
        fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> c_int;
        /// The device's name, NUL-terminated, in at most `len` bytes.
        fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CuDevice) -> c_int;
        /// The device's memory in bytes.
        fn cuDeviceTotalMem_v2(bytes: *mut usize, device: CuDevice) -> c_int;
        /// One numeric attribute of the device.
        fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CuDevice) -> c_int;
        /// Takes a reference on the device's primary context, creating it if needed.
        fn cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice) -> c_int;
        /// Gives back a reference on the device's primary context.
        fn cuDevicePrimaryCtxRelease_v2(device: CuDevice) -> c_int;
        /// Makes `context` the calling thread's current one.
        fn cuCtxSetCurrent(context: CuContext) -> c_int;
        /// Waits until the current context's submitted work is done.
        fn cuCtxSynchronize() -> c_int;
        /// The current context's device memory, free and in all, in bytes.
        fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
        /// Allocates device memory.
        fn cuMemAlloc_v2(pointer: *mut DevicePtr, bytes: usize) -> c_int;
        /// Frees device memory from `cuMemAlloc_v2`.
        fn cuMemFree_v2(pointer: DevicePtr) -> c_int;
        /// Sets `count` bytes of device memory to `value`.
        fn cuMemsetD8_v2(to: DevicePtr, value: u8, count: usize) -> c_int;
        /// Copies host memory to device memory, after the context's earlier work.
        fn cuMemcpyHtoD_v2(to: DevicePtr, from: *const c_void, bytes: usize) -> c_int;
        /// Copies device memory to host memory, after the context's earlier work.
        fn cuMemcpyDtoH_v2(to: *mut c_void, from: DevicePtr, bytes: usize) -> c_int;
        /// Loads a module from a NUL-terminated PTX text.
        fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> c_int;
        /// Unloads a module.
        fn cuModuleUnload(module: CuModule) -> c_int;
        /// The kernel a module holds under `name`.
        fn cuModuleGetFunction(function: *mut CuFunction, module: CuModule, name: *const c_char) -> c_int;
        /// Starts a kernel on a grid of blocks; `params` points to one pointer per kernel
        /// parameter, each to that parameter's value.
        fn cuLaunchKernel(
            function: CuFunction,
            grid_x: c_uint,
            grid_y: c_uint,
            grid_z: c_uint,
            block_x: c_uint,
            block_y: c_uint,
            block_z: c_uint,
            shared_bytes: c_uint,
            stream: CuStream,
            params: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> c_int;
        /// The name of a status code, as a static NUL-terminated string.
        fn cuGetErrorName(code: c_int, name: *mut *const c_char) -> c_int;
    }
}

api! {
    /// cuDNN, NVIDIA's library of neural-network primitives.
    Cudnn, "system" {
        /// The library's version, as `10000 * major + 100 * minor + patch`.
        fn cudnnGetVersion() -> usize;
        /// The CUDA version the library was built for, as `1000 * major + 10 * minor`.
        fn cudnnGetCudartVersion() -> usize;
        /// Creates a handle in the calling thread's current context.
        fn cudnnCreate(handle: *mut *mut c_void) -> c_int;
        /// Destroys a handle.
        fn cudnnDestroy(handle: *mut c_void) -> c_int;
        /// The text of a status code, as a static NUL-terminated string.
        fn cudnnGetErrorString(status: c_int) -> *const c_char;
    }
}

api! {
    /// cuBLAS, NVIDIA's linear algebra library.
    Cublas, "system" {
        /// Creates a handle in the calling thread's current context.
        fn cublasCreate_v2(handle: *mut *mut c_void) -> c_int;
        /// Destroys a handle.
        fn cublasDestroy_v2(handle: *mut c_void) -> c_int;
        /// The library's version, as `10000 * major + 100 * minor + patch`.
        fn cublasGetVersion_v2(handle: *mut c_void, version: *mut c_int) -> c_int;
        /// The text of a status code, as a static NUL-terminated string.
        fn cublasGetStatusString(status: c_int) -> *const c_char;
    }
}

api! {
    /// NVRTC, NVIDIA's run-time compiler from CUDA C++ to PTX.
    Nvrtc, "C" {
        /// The compiler's version.
        fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> c_int;
        /// Makes a program from one source text; no headers are passed.
        fn nvrtcCreateProgram(
            program: *mut *mut c_void,
            source: *const c_char,
            name: *const c_char,
            headers: c_int,
            header_sources: *const *const c_char,
            include_names: *const *const c_char,
        ) -> c_int;
        /// Compiles a program with the given options.
        fn nvrtcCompileProgram(program: *mut c_void, options: c_int, option_list: *const *const c_char) -> c_int;
        /// The size of the compiled PTX, its terminating NUL included.
        fn nvrtcGetPTXSize(program: *mut c_void, size: *mut usize) -> c_int;
        /// Copies the compiled PTX, NUL-terminated.
        fn nvrtcGetPTX(program: *mut c_void, ptx: *mut c_char) -> c_int;
        /// The size of the compiler's log, its terminating NUL included.
        fn nvrtcGetProgramLogSize(program: *mut c_void, size: *mut usize) -> c_int;
        /// Copies the compiler's log, NUL-terminated.
        fn nvrtcGetProgramLog(program: *mut c_void, log: *mut c_char) -> c_int;
        /// Destroys a program and nulls the pointer.
        fn nvrtcDestroyProgram(program: *mut *mut c_void) -> c_int;
        /// The text of a status code, as a static NUL-terminated string.
        fn nvrtcGetErrorString(code: c_int) -> *const c_char;
    }
}
