//! Opening CUDA: the driver, a card, cuDNN, cuBLAS and NVRTC, loaded at run time, with a
//! context on the card and a handle from each library.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::CudaError;
use crate::find::{
    CUDA_MAJORS, CUDNN_MAJOR, Platform, cublas_name, cublaslt_name, cudnn_name, cudnn_parts,
    driver_name, is_nvrtc_builtins, locate, nvrtc_name, search_dirs, siblings,
};
use crate::library::Library;
use crate::sys::{
    COMPUTE_CAPABILITY_MAJOR, COMPUTE_CAPABILITY_MINOR, CuContext, CuDevice, Cublas, Cudnn, Driver,
    ERROR_NO_DEVICE, Nvrtc,
};

/// The environment variable that turns "CUDA cannot be used" into a panic in
/// [`Cuda::new`], for test runs that must use it.
const REQUIRED_VAR: &str = "CAVITATION_CUDA_REQUIRED";

/// What a [`Cuda`] runs on, as [`Cuda::info`] gives it.
///
/// Its [`Display`](fmt::Display) form is a few lines for a log or a benchmark's header.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::Cuda;
///
/// if let Some(cuda) = Cuda::new() {
///     let info = cuda.info();
///     assert_eq!(info.cudnn.0, 9);
///     println!("{info}");
/// }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CudaInfo {
    /// The card's name, such as "NVIDIA GeForce GTX 1660 SUPER".
    pub name: String,
    /// The card's memory in bytes.
    pub memory: u64,
    /// The card's compute capability, major and minor: (7, 5) for a GTX 1660 Super.
    pub compute_capability: (u32, u32),
    /// The newest CUDA version the installed driver supports, major and minor.
    pub driver_cuda: (u32, u32),
    /// cuDNN's version: major, minor and patch.
    pub cudnn: (u32, u32, u32),
    /// The CUDA version cuDNN was built for, major and minor. cuBLAS and NVRTC are loaded
    /// in the same major version.
    pub cudnn_cuda: (u32, u32),
    /// cuBLAS's version: major, minor and patch.
    pub cublas: (u32, u32, u32),
    /// NVRTC's version, major and minor.
    pub nvrtc: (u32, u32),
    /// The compute capability kernels are compiled for: the card's, or the newest older
    /// one NVRTC compiles for, which the card also runs.
    pub compiles_for: (u32, u32),
    /// The files cuDNN, cuBLAS and NVRTC were loaded from, in that order; a bare file name
    /// for one the operating system's linker found in its own folders.
    pub libraries: Vec<PathBuf>,
}

impl fmt::Display for CudaInfo {
    /// Writes the card, the versions and the files loaded, one per line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        writeln!(
            f,
            "{}, {:.1} GiB, compute capability {}.{}",
            self.name,
            self.memory as f64 / GIB,
            self.compute_capability.0,
            self.compute_capability.1
        )?;
        writeln!(
            f,
            "driver supports CUDA {}.{}; cuDNN {}.{}.{} (built for CUDA {}.{}), cuBLAS \
             {}.{}.{}, NVRTC {}.{} (compiling for {}.{})",
            self.driver_cuda.0,
            self.driver_cuda.1,
            self.cudnn.0,
            self.cudnn.1,
            self.cudnn.2,
            self.cudnn_cuda.0,
            self.cudnn_cuda.1,
            self.cublas.0,
            self.cublas.1,
            self.cublas.2,
            self.nvrtc.0,
            self.nvrtc.1,
            self.compiles_for.0,
            self.compiles_for.1
        )?;
        let files: Vec<String> = self
            .libraries
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        write!(f, "loaded: {}", files.join(", "))
    }
}

/// CUDA on one NVIDIA card: the driver, a context on the card, and cuDNN, cuBLAS and NVRTC
/// with a handle from each.
///
/// The libraries are found and loaded at run time, so a program built with this crate
/// starts on a machine that has none of them, and [`Cuda::open`] says what is missing.
/// The card is the driver's first; the context is the card's primary context, which other
/// CUDA code in the process shares.
///
/// A `Cuda` belongs to the thread that opened it, so the type is neither `Send` nor `Sync`.
/// Buffers and modules borrow it and are dropped before it. Each call makes the card's
/// context current on the thread first, so other CUDA code on the same thread may change
/// the current context between calls; when the `Cuda` is dropped, the thread is left with
/// no current context. Two `Cuda`s on one thread share the card's primary context, so a
/// buffer of one may be passed to a kernel of the other.
///
/// ```compile_fail
/// fn send<T: Send>(_: T) {}
/// send(cavitation_cuda::Cuda::open().unwrap());
/// ```
///
/// ```compile_fail
/// fn share<T: Sync>(_: &T) {}
/// share(&cavitation_cuda::Cuda::open().unwrap());
/// ```
///
/// # Examples
///
/// ```
/// use cavitation_cuda::Cuda;
///
/// match Cuda::open() {
///     Ok(cuda) => println!("{}", cuda.info()),
///     Err(why) => println!("CUDA is not available: {why}"),
/// }
/// ```
pub struct Cuda {
    /// The driver API's functions.
    pub(crate) driver: Driver,
    /// cuDNN's functions.
    cudnn: Cudnn,
    /// cuBLAS's functions.
    cublas: Cublas,
    /// NVRTC's functions.
    pub(crate) nvrtc: Nvrtc,
    /// The card.
    device: CuDevice,
    /// The card's primary context, retained once and current on the opening thread; null
    /// until it is retained.
    context: CuContext,
    /// The cuDNN handle, or null until it is created.
    cudnn_handle: *mut c_void,
    /// The cuBLAS handle, or null until it is created.
    cublas_handle: *mut c_void,
    /// What was opened.
    info: CudaInfo,
}

impl fmt::Debug for Cuda {
    /// Writes the card's name and the library versions.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cuda").field("info", &self.info).finish()
    }
}

impl Cuda {
    /// Opens CUDA, or returns `None` when it cannot be used here: no NVIDIA card or driver,
    /// or a library that is not installed. [`Cuda::open`] gives the reason.
    ///
    /// # Panics
    ///
    /// If CUDA cannot be used while `CAVITATION_CUDA_REQUIRED` is `1`, with the reason.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     println!("{}", cuda.info().name);
    /// }
    /// ```
    pub fn new() -> Option<Cuda> {
        match Cuda::open() {
            Ok(cuda) => Some(cuda),
            Err(why) => {
                assert!(
                    !required_from_env(),
                    "{REQUIRED_VAR} is 1, but CUDA cannot be used: {why}"
                );
                None
            }
        }
    }

    /// Opens CUDA, and says why when it cannot be used.
    ///
    /// The steps, each a reason of its own when it fails:
    ///
    /// 1. The driver library (`nvcuda.dll` from the Windows system folder, `libcuda.so.1`
    ///    on Linux) is loaded and asked for its first card.
    /// 2. cuDNN 9 is looked for in the folders of `CAVITATION_CUDA_PATH`, `CUDNN_PATH` and
    ///    `CUDA_PATH` (each with its `bin`, `bin/x64`, `lib`, `lib64` and `lib/x64`), then
    ///    in the folders of `PATH` (Windows) or `LD_LIBRARY_PATH` and the linker's own
    ///    (Linux). Its parts beside it are loaded with it.
    /// 3. cuBLAS and NVRTC are looked for in the same folders, in the CUDA major version
    ///    cuDNN was built for.
    /// 4. The card's primary context is made current on this thread, and cuDNN and cuBLAS
    ///    each give a handle in it.
    ///
    /// Libraries are loaded by absolute path, or by the linker from its configured folders,
    /// and neither these nor the libraries they import are looked for in the folder the
    /// program was started in. A library NVIDIA's code loads by name later, while it runs,
    /// is looked for as the operating system does by default; the ones known (cuDNN's parts,
    /// cuBLASLt, NVRTC's built-ins and, on Windows, `zlibwapi.dll`) are loaded here first
    /// from beside the library that needs them. Once loaded they stay loaded until the
    /// process ends.
    ///
    /// `CAVITATION_CUDA_REQUIRED` does not apply here: it makes [`Cuda::new`] panic.
    ///
    /// # Errors
    ///
    /// A [`CudaError`] naming the step that failed: [`Unsupported`](CudaError::Unsupported)
    /// on an operating system without CUDA, [`NoDriver`](CudaError::NoDriver),
    /// [`NoDevice`](CudaError::NoDevice), [`Missing`](CudaError::Missing) with every library
    /// that was not found, [`Version`](CudaError::Version) for a cuDNN that is not version
    /// 9, and [`Symbol`](CudaError::Symbol) or [`Call`](CudaError::Call) when a library is
    /// there and does not work, as with a driver older than the libraries need.
    pub fn open() -> Result<Cuda, CudaError> {
        let platform = Platform::here();
        if platform == Platform::Other {
            return Err(CudaError::Unsupported);
        }
        let search = search_dirs(platform, |name| std::env::var_os(name));
        let dirs = search.all();

        let driver = open_driver(platform, &dirs)?;
        let device = first_device(&driver)?;

        let mut problems = Vec::new();
        let cudnn = match load_cudnn(platform, &dirs) {
            Ok(found) => Some(found),
            Err(LoadError::Missing(problem)) => {
                problems.push(problem);
                None
            }
            Err(LoadError::Fatal(error)) => return Err(error),
        };
        // With cuDNN loaded, the CUDA major version is the one it was built for; without
        // it, either is looked for, so that the error names everything that is missing.
        let majors: Vec<u32> = match &cudnn {
            Some(found) if !CUDA_MAJORS.contains(&found.cuda.0) => {
                return Err(CudaError::Version(format!(
                    "{} is cuDNN {}.{}.{} built for CUDA {}.{}, and this library uses cuDNN \
                     {CUDNN_MAJOR} built for CUDA 12 or 13: install that build",
                    found.path.display(),
                    found.version.0,
                    found.version.1,
                    found.version.2,
                    found.cuda.0,
                    found.cuda.1
                )));
            }
            Some(found) => vec![found.cuda.0],
            None => CUDA_MAJORS.to_vec(),
        };
        // cuBLAS imports cuBLASLt, which is loaded first from beside it when it is there.
        let cublas = load_versioned(
            platform,
            &dirs,
            &majors,
            "cuBLAS",
            cublas_name,
            |major, path| {
                let lt = path.with_file_name(cublaslt_name(platform, major));
                if lt.is_file() { vec![lt] } else { Vec::new() }
            },
        );
        let nvrtc = load_versioned(
            platform,
            &dirs,
            &majors,
            "NVRTC",
            nvrtc_name,
            |major, path| siblings(path, |file| is_nvrtc_builtins(platform, major, file)),
        );
        let (cudnn, cublas, nvrtc) = match (cudnn, cublas, nvrtc) {
            (Some(cudnn), Ok(cublas), Ok(nvrtc)) => (cudnn, cublas, nvrtc),
            (_, cublas, nvrtc) => {
                problems.extend(cublas.err());
                problems.extend(nvrtc.err());
                return Err(CudaError::Missing {
                    problems,
                    searched: search.named,
                    loader_folders: search.loader,
                });
            }
        };
        let symbol = |library: &Path, symbol| CudaError::Symbol {
            library: library.display().to_string(),
            symbol,
        };
        let cublas_table = Cublas::load(&cublas.0).map_err(|s| symbol(&cublas.1, s))?;
        let nvrtc_table = Nvrtc::load(&nvrtc.0).map_err(|s| symbol(&nvrtc.1, s))?;

        let name = device_name(&driver, device)?;
        let mut memory = 0usize;
        // SAFETY: the pointer is to a live `usize`, which the function writes once.
        let code = unsafe { (driver.cuDeviceTotalMem_v2)(&mut memory, device) };
        driver_check(&driver, "cuDeviceTotalMem_v2", code)?;
        let compute_capability = (
            device_attribute(&driver, device, COMPUTE_CAPABILITY_MAJOR)?,
            device_attribute(&driver, device, COMPUTE_CAPABILITY_MINOR)?,
        );
        let mut version: c_int = 0;
        // SAFETY: the pointer is to a live `c_int`, which the function writes once.
        let code = unsafe { (driver.cuDriverGetVersion)(&mut version) };
        driver_check(&driver, "cuDriverGetVersion", code)?;
        let version = u32::try_from(version).unwrap_or(0);
        let (mut major, mut minor): (c_int, c_int) = (0, 0);
        // SAFETY: both pointers are to live `c_int`s, each written once.
        let code = unsafe { (nvrtc_table.nvrtcVersion)(&mut major, &mut minor) };
        if code != 0 {
            return Err(nvrtc_error(&nvrtc_table, "nvrtcVersion", code));
        }
        let nvrtc_version = (
            u32::try_from(major).unwrap_or(0),
            u32::try_from(minor).unwrap_or(0),
        );
        let driver_cuda = (version / 1000, version % 1000 / 10);
        // The driver compiles the PTX NVRTC writes, and a driver older than NVRTC does not
        // read its PTX.
        check_driver_reads(driver_cuda, nvrtc_version)?;
        let card = compute_capability.0 * 10 + compute_capability.1;
        let archs = supported_archs(&nvrtc_table)?;
        let Some(arch) = pick_arch(card, &archs) else {
            return Err(CudaError::Version(format!(
                "the card's compute capability {}.{} is older than any NVRTC {}.{} compiles \
                 for (from {})",
                compute_capability.0,
                compute_capability.1,
                nvrtc_version.0,
                nvrtc_version.1,
                archs
                    .first()
                    .map_or("none".to_string(), |a| format!("{}.{}", a / 10, a % 10))
            )));
        };

        let mut cuda = Cuda {
            driver,
            cudnn: cudnn.table,
            cublas: cublas_table,
            nvrtc: nvrtc_table,
            device,
            context: std::ptr::null_mut(),
            cudnn_handle: std::ptr::null_mut(),
            cublas_handle: std::ptr::null_mut(),
            info: CudaInfo {
                name,
                memory: memory as u64,
                compute_capability,
                driver_cuda,
                cudnn: cudnn.version,
                cudnn_cuda: cudnn.cuda,
                cublas: (0, 0, 0),
                nvrtc: nvrtc_version,
                compiles_for: (arch / 10, arch % 10),
                libraries: vec![cudnn.path, cublas.1, nvrtc.1],
            },
        };
        // From here a failure drops `cuda`, which gives back whatever was taken so far.
        cuda.start()?;
        Ok(cuda)
    }

    /// Retains the card's primary context, makes it current on this thread, and creates
    /// the cuDNN and cuBLAS handles in it.
    fn start(&mut self) -> Result<(), CudaError> {
        // SAFETY: the pointer is to this struct's context field, written once, and the
        // device came from `cuDeviceGet`.
        let code =
            unsafe { (self.driver.cuDevicePrimaryCtxRetain)(&mut self.context, self.device) };
        if code != 0 {
            self.context = std::ptr::null_mut();
        }
        self.check("cuDevicePrimaryCtxRetain", code)?;
        // SAFETY: the context was just retained and is released only when `self` drops.
        let code = unsafe { (self.driver.cuCtxSetCurrent)(self.context) };
        self.check("cuCtxSetCurrent", code)?;

        // SAFETY: the pointer is to this struct's handle field, written once, and a
        // context is current on this thread.
        let status = unsafe { (self.cudnn.cudnnCreate)(&mut self.cudnn_handle) };
        if status != 0 {
            self.cudnn_handle = std::ptr::null_mut();
            return Err(self.cudnn_error("cudnnCreate", status));
        }
        // SAFETY: as for cuDNN's handle above.
        let status = unsafe { (self.cublas.cublasCreate_v2)(&mut self.cublas_handle) };
        if status != 0 {
            self.cublas_handle = std::ptr::null_mut();
            return Err(self.cublas_error("cublasCreate_v2", status));
        }
        let mut version: c_int = 0;
        // SAFETY: the handle was just created, and the pointer is to a live `c_int`.
        let status = unsafe { (self.cublas.cublasGetVersion_v2)(self.cublas_handle, &mut version) };
        if status != 0 {
            return Err(self.cublas_error("cublasGetVersion_v2", status));
        }
        let version = u32::try_from(version).unwrap_or(0);
        self.info.cublas = (version / 10000, version / 100 % 100, version % 100);
        Ok(())
    }

    /// What was opened: the card, the library versions and the files they came from.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     assert_eq!(cuda.info().libraries.len(), 3);
    /// }
    /// ```
    pub fn info(&self) -> &CudaInfo {
        &self.info
    }

    /// The card's free memory in bytes, as the driver counts it now. Other programs using
    /// the card change it.
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
    ///     assert!(cuda.free_memory().unwrap() <= cuda.info().memory);
    /// }
    /// ```
    pub fn free_memory(&self) -> Result<u64, CudaError> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both pointers are to live `usize`s, each written once, and this
        // thread's current context is the one opened.
        let code = unsafe { (self.driver.cuMemGetInfo_v2)(&mut free, &mut total) };
        self.check("cuMemGetInfo_v2", code)?;
        Ok(free as u64)
    }

    /// Waits until every kernel launched and every copy started on this card's context is
    /// done.
    ///
    /// # Errors
    ///
    /// [`CudaError::Call`] if the driver reports an error, which is also how a kernel that
    /// failed after its launch returned is reported.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_cuda::Cuda;
    ///
    /// if let Some(cuda) = Cuda::new() {
    ///     cuda.synchronize().unwrap();
    /// }
    /// ```
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.bind()?;
        // SAFETY: the function takes no arguments and works on this thread's current
        // context, the one opened.
        let code = unsafe { (self.driver.cuCtxSynchronize)() };
        self.check("cuCtxSynchronize", code)
    }

    /// Makes the card's context current on the calling thread, which is the thread that
    /// opened it (the type is not `Send`), so that the driver call after it works on this
    /// card whatever other code made current in between.
    pub(crate) fn bind(&self) -> Result<(), CudaError> {
        // SAFETY: the context is retained until `self` drops.
        let code = unsafe { (self.driver.cuCtxSetCurrent)(self.context) };
        self.check("cuCtxSetCurrent", code)
    }

    /// The compute architecture kernels are compiled for, as `10 * major + minor`: the
    /// card's, or the newest older one NVRTC knows.
    pub(crate) fn arch(&self) -> u32 {
        self.info.compiles_for.0 * 10 + self.info.compiles_for.1
    }

    /// `Ok` for a driver status of 0, and the driver's name for any other.
    pub(crate) fn check(&self, function: &'static str, code: c_int) -> Result<(), CudaError> {
        driver_check(&self.driver, function, code)
    }

    /// The error for a cuDNN status.
    fn cudnn_error(&self, function: &'static str, status: c_int) -> CudaError {
        // SAFETY: the function takes any status and returns a static string.
        let text = unsafe { (self.cudnn.cudnnGetErrorString)(status) };
        CudaError::Call {
            function,
            code: status,
            message: static_text(text),
        }
    }

    /// The error for a cuBLAS status.
    fn cublas_error(&self, function: &'static str, status: c_int) -> CudaError {
        // SAFETY: the function takes any status and returns a static string.
        let text = unsafe { (self.cublas.cublasGetStatusString)(status) };
        CudaError::Call {
            function,
            code: status,
            message: static_text(text),
        }
    }

    /// The error for an NVRTC status.
    pub(crate) fn nvrtc_error(&self, function: &'static str, code: c_int) -> CudaError {
        nvrtc_error(&self.nvrtc, function, code)
    }
}

impl Drop for Cuda {
    /// Waits for the card's work, destroys the handles, gives the context back and leaves
    /// the thread with no current context. Errors are not reported: there is no one to
    /// report them to.
    fn drop(&mut self) {
        if self.context.is_null() {
            return;
        }
        // SAFETY: the context is retained and is made current on this thread first, each
        // handle is destroyed once and only if it was created, the context's reference is
        // the one `start` took, and a null context is a valid argument to `cuCtxSetCurrent`.
        unsafe {
            (self.driver.cuCtxSetCurrent)(self.context);
            (self.driver.cuCtxSynchronize)();
            if !self.cudnn_handle.is_null() {
                (self.cudnn.cudnnDestroy)(self.cudnn_handle);
            }
            if !self.cublas_handle.is_null() {
                (self.cublas.cublasDestroy_v2)(self.cublas_handle);
            }
            (self.driver.cuDevicePrimaryCtxRelease_v2)(self.device);
            (self.driver.cuCtxSetCurrent)(std::ptr::null_mut());
        }
    }
}

/// cuDNN, loaded and asked for its versions.
struct LoadedCudnn {
    /// Its functions.
    table: Cudnn,
    /// The file it was loaded from.
    path: PathBuf,
    /// Its version: major, minor and patch.
    version: (u32, u32, u32),
    /// The CUDA version it was built for, major and minor.
    cuda: (u32, u32),
}

/// Why a library was not loaded.
enum LoadError {
    /// It was not found or would not load: one sentence for [`CudaError::Missing`].
    Missing(String),
    /// It loaded and cannot be used: the error to return as it is.
    Fatal(CudaError),
}

/// Whether `CAVITATION_CUDA_REQUIRED` makes CUDA required.
fn required_from_env() -> bool {
    parse_required(std::env::var(REQUIRED_VAR).ok().as_deref())
}

/// Reads `CAVITATION_CUDA_REQUIRED`'s value: only `1` (with surrounding spaces allowed)
/// makes CUDA required.
fn parse_required(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Copies a static NUL-terminated string a library returned, or says there was none.
fn static_text(text: *const c_char) -> String {
    if text.is_null() {
        return "unknown status".to_string();
    }
    // SAFETY: non-null, and the libraries return pointers to static NUL-terminated
    // strings from their status-name functions.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// `Ok` for a driver status of 0, and the driver's name for any other.
fn driver_check(driver: &Driver, function: &'static str, code: c_int) -> Result<(), CudaError> {
    if code == 0 {
        return Ok(());
    }
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: the pointer is to a live pointer, which the function sets to a static
    // string or leaves null for a status it does not know.
    unsafe { (driver.cuGetErrorName)(code, &mut name) };
    Err(CudaError::Call {
        function,
        code,
        message: static_text(name),
    })
}

/// The error for an NVRTC status.
fn nvrtc_error(nvrtc: &Nvrtc, function: &'static str, code: c_int) -> CudaError {
    // SAFETY: the function takes any status and returns a static string.
    let text = unsafe { (nvrtc.nvrtcGetErrorString)(code) };
    CudaError::Call {
        function,
        code,
        message: static_text(text),
    }
}

/// Why [`load`] did not load a library.
#[derive(Debug, PartialEq, Eq)]
enum LoadFailure {
    /// No folder holds the file.
    NotFound(String),
    /// The file, or one loaded before it, was found and would not load: one sentence.
    Failed(String),
}

impl LoadFailure {
    /// The failure as one sentence.
    fn text(self) -> String {
        match self {
            LoadFailure::NotFound(name) => format!("{name} was not found"),
            LoadFailure::Failed(why) => why,
        }
    }
}

/// Loads the library called `name`: from the first of `dirs` that holds it, after the
/// files `before` lists for that path, or, on Linux, from the linker's own folders.
///
/// # Errors
///
/// [`LoadFailure::NotFound`], or [`LoadFailure::Failed`] when it or a file loaded before it
/// would not load.
fn load(
    platform: Platform,
    dirs: &[PathBuf],
    name: &str,
    before: impl Fn(&Path) -> Vec<PathBuf>,
) -> Result<(Library, PathBuf), LoadFailure> {
    let failed = |path: &Path, why: String| {
        LoadFailure::Failed(format!("{} could not be loaded: {why}", path.display()))
    };
    if let Some(path) = locate(dirs, name) {
        for other in before(&path) {
            Library::open(&other).map_err(|why| failed(&other, why))?;
        }
        return match Library::open(&path) {
            Ok(library) => Ok((library, path)),
            Err(why) => Err(failed(&path, why)),
        };
    }
    if platform == Platform::Linux {
        return match Library::open_system(name) {
            Ok(library) => Ok((library, PathBuf::from(name))),
            Err(why) => Err(system_failure(name, why)),
        };
    }
    Err(LoadFailure::NotFound(name.to_string()))
}

/// What a failed load by bare name through the linker means: not found when the linker
/// says it found no such file, and its own message otherwise (a library that is there and
/// does not load, or a folder of `LD_LIBRARY_PATH` refused as relative).
fn system_failure(name: &str, why: String) -> LoadFailure {
    if why.contains("No such file or directory") {
        LoadFailure::NotFound(name.to_string())
    } else {
        LoadFailure::Failed(format!("{name} could not be loaded: {why}"))
    }
}

/// `Ok` when the driver reads the PTX NVRTC writes, which it does when it supports at
/// least NVRTC's CUDA version, both as major and minor.
fn check_driver_reads(driver_cuda: (u32, u32), nvrtc: (u32, u32)) -> Result<(), CudaError> {
    if driver_cuda < nvrtc {
        return Err(CudaError::Version(format!(
            "the NVIDIA driver supports CUDA {}.{}, older than NVRTC {}.{}: update the driver",
            driver_cuda.0, driver_cuda.1, nvrtc.0, nvrtc.1
        )));
    }
    Ok(())
}

/// The compute architectures NVRTC compiles for, as `10 * major + minor`, oldest first.
fn supported_archs(nvrtc: &Nvrtc) -> Result<Vec<u32>, CudaError> {
    let mut count: c_int = 0;
    // SAFETY: the pointer is to a live `c_int`, which the function writes once.
    let code = unsafe { (nvrtc.nvrtcGetNumSupportedArchs)(&mut count) };
    if code != 0 {
        return Err(nvrtc_error(nvrtc, "nvrtcGetNumSupportedArchs", code));
    }
    let mut archs: Vec<c_int> = vec![0; usize::try_from(count).unwrap_or(0)];
    if !archs.is_empty() {
        // SAFETY: the buffer holds as many `c_int`s as the function just counted.
        let code = unsafe { (nvrtc.nvrtcGetSupportedArchs)(archs.as_mut_ptr()) };
        if code != 0 {
            return Err(nvrtc_error(nvrtc, "nvrtcGetSupportedArchs", code));
        }
    }
    let mut archs: Vec<u32> = archs
        .into_iter()
        .filter_map(|a| u32::try_from(a).ok())
        .collect();
    archs.sort_unstable();
    Ok(archs)
}

/// The architecture to compile for on a card of compute capability `card` (as
/// `10 * major + minor`): the newest of `archs` that is not newer than the card, whose PTX
/// the card runs. `None` when every one is newer than the card.
fn pick_arch(card: u32, archs: &[u32]) -> Option<u32> {
    archs.iter().copied().filter(|&a| a <= card).max()
}

/// Loads the driver library and its functions, and initialises the driver API.
fn open_driver(platform: Platform, dirs: &[PathBuf]) -> Result<Driver, CudaError> {
    let name = driver_name(platform);
    let no_driver = |message: String| CudaError::NoDriver {
        library: name.to_string(),
        message,
    };
    let library = if platform == Platform::Windows {
        // The display driver installs it in the system folder, and nowhere else is tried.
        Library::open_system(name).map_err(no_driver)?
    } else {
        load(platform, dirs, name, |_| Vec::new())
            .map_err(|failure| no_driver(failure.text()))?
            .0
    };
    let driver = Driver::load(&library).map_err(|symbol| CudaError::Symbol {
        library: name.to_string(),
        symbol,
    })?;
    // SAFETY: the flags must be 0, and the function may be called more than once.
    let code = unsafe { (driver.cuInit)(0) };
    if code == ERROR_NO_DEVICE {
        return Err(CudaError::NoDevice);
    }
    driver_check(&driver, "cuInit", code)?;
    Ok(driver)
}

/// The driver's first card.
fn first_device(driver: &Driver) -> Result<CuDevice, CudaError> {
    let mut count: c_int = 0;
    // SAFETY: the pointer is to a live `c_int`, which the function writes once.
    let code = unsafe { (driver.cuDeviceGetCount)(&mut count) };
    driver_check(driver, "cuDeviceGetCount", code)?;
    if count < 1 {
        return Err(CudaError::NoDevice);
    }
    let mut device: CuDevice = 0;
    // SAFETY: the pointer is to a live handle, and ordinal 0 is below the count.
    let code = unsafe { (driver.cuDeviceGet)(&mut device, 0) };
    driver_check(driver, "cuDeviceGet", code)?;
    Ok(device)
}

/// The card's name.
fn device_name(driver: &Driver, device: CuDevice) -> Result<String, CudaError> {
    let mut name = [0 as c_char; 256];
    // SAFETY: the buffer is 256 bytes, the length passed, and the function writes a
    // NUL-terminated string within it.
    let code = unsafe { (driver.cuDeviceGetName)(name.as_mut_ptr(), 256, device) };
    driver_check(driver, "cuDeviceGetName", code)?;
    name[255] = 0;
    // SAFETY: the buffer holds a NUL within its length, set above if nowhere earlier.
    let name = unsafe { CStr::from_ptr(name.as_ptr()) };
    Ok(name.to_string_lossy().into_owned())
}

/// One numeric attribute of the card.
fn device_attribute(driver: &Driver, device: CuDevice, attribute: c_int) -> Result<u32, CudaError> {
    let mut value: c_int = 0;
    // SAFETY: the pointer is to a live `c_int`, which the function writes once.
    let code = unsafe { (driver.cuDeviceGetAttribute)(&mut value, attribute, device) };
    driver_check(driver, "cuDeviceGetAttribute", code)?;
    Ok(u32::try_from(value).unwrap_or(0))
}

/// The zlib build NVIDIA's Windows libraries name, loaded by cuDNN by name when it is not
/// loaded already.
const ZLIB: &str = "zlibwapi.dll";

/// Finds and loads cuDNN with the parts beside it, and reads its versions.
fn load_cudnn(platform: Platform, dirs: &[PathBuf]) -> Result<LoadedCudnn, LoadError> {
    let name = cudnn_name(platform);
    // cuDNN loads its parts by bare name when first called. Loading them here by path,
    // before it, is what lets it find them in a folder the loader is not pointed at.
    let (library, path) = load(platform, dirs, &name, |path| {
        let mut first = cudnn_parts(platform, path);
        let zlib = path.with_file_name(ZLIB);
        if platform == Platform::Windows && zlib.is_file() {
            first.insert(0, zlib);
        }
        first
    })
    .map_err(|failure| LoadError::Missing(format!("cuDNN {CUDNN_MAJOR}: {}", failure.text())))?;
    let table = Cudnn::load(&library).map_err(|symbol| {
        LoadError::Fatal(CudaError::Symbol {
            library: path.display().to_string(),
            symbol,
        })
    })?;
    // SAFETY: both functions take no arguments and return a number.
    let (version, cuda) = unsafe { ((table.cudnnGetVersion)(), (table.cudnnGetCudartVersion)()) };
    let narrow = |v: usize| u32::try_from(v).unwrap_or(0);
    let version = (
        narrow(version / 10000),
        narrow(version / 100 % 100),
        narrow(version % 100),
    );
    let cuda = (narrow(cuda / 1000), narrow(cuda % 1000 / 10));
    if version.0 != CUDNN_MAJOR {
        return Err(LoadError::Fatal(CudaError::Version(format!(
            "{} is cuDNN {}.{}.{}, and this library calls cuDNN {CUDNN_MAJOR}",
            path.display(),
            version.0,
            version.1,
            version.2
        ))));
    }
    Ok(LoadedCudnn {
        table,
        path,
        version,
        cuda,
    })
}

/// Finds and loads a library whose file name carries the CUDA major version, trying
/// `majors` in order; `before` lists the files to load first, given the major version and
/// the library's path.
///
/// # Errors
///
/// One sentence for [`CudaError::Missing`]: the file names tried, or the file that would
/// not load.
fn load_versioned(
    platform: Platform,
    dirs: &[PathBuf],
    majors: &[u32],
    what: &str,
    name: impl Fn(Platform, u32) -> String,
    before: impl Fn(u32, &Path) -> Vec<PathBuf>,
) -> Result<(Library, PathBuf), String> {
    let mut tried = Vec::new();
    for &major in majors {
        let file = name(platform, major);
        match load(platform, dirs, &file, |path| before(major, path)) {
            Ok(found) => return Ok(found),
            Err(LoadFailure::NotFound(_)) => tried.push(file),
            Err(LoadFailure::Failed(why)) => return Err(format!("{what}: {why}")),
        }
    }
    Err(format!("{what}: {} was not found", tried.join(" or ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `1` makes CUDA required.
    #[test]
    fn only_one_requires() {
        assert!(parse_required(Some("1")));
        assert!(parse_required(Some(" 1 ")));
        assert!(!parse_required(Some("0")));
        assert!(!parse_required(Some("true")));
        assert!(!parse_required(Some("")));
        assert!(!parse_required(None));
    }

    /// With no folder to search, a versioned library's error names each file name tried.
    #[test]
    fn a_library_not_found_names_the_files_tried() {
        // Windows names, so that the Linux linker's own folders are not asked.
        let err = load_versioned(
            Platform::Windows,
            &[],
            &CUDA_MAJORS,
            "cuBLAS",
            cublas_name,
            |_, _| Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            "cuBLAS: cublas64_13.dll or cublas64_12.dll was not found"
        );
    }

    /// The architecture is the card's when NVRTC has it, else the newest older one, else
    /// none.
    #[test]
    fn the_architecture_is_the_cards_or_older() {
        let archs = [75, 80, 86, 89, 90, 100, 120];
        assert_eq!(pick_arch(75, &archs), Some(75));
        assert_eq!(pick_arch(87, &archs), Some(86));
        assert_eq!(pick_arch(121, &archs), Some(120));
        assert_eq!(pick_arch(130, &[75, 120]), Some(120));
        assert_eq!(pick_arch(61, &archs), None);
        assert_eq!(pick_arch(75, &[]), None);
    }

    /// The linker's message decides between not found and found but not loaded.
    #[test]
    fn the_linkers_message_is_kept() {
        let missing = "libcudnn.so.9: cannot open shared object file: No such file or directory";
        assert_eq!(
            system_failure("libcudnn.so.9", missing.to_string()),
            LoadFailure::NotFound("libcudnn.so.9".to_string())
        );
        let refused = "LD_LIBRARY_PATH holds the folder \"\", which the linker would read";
        assert_eq!(
            system_failure("libcuda.so.1", refused.to_string()).text(),
            format!("libcuda.so.1 could not be loaded: {refused}")
        );
    }

    /// A driver reads NVRTC's PTX when it supports NVRTC's CUDA version or a newer one.
    #[test]
    fn the_driver_must_be_as_new_as_nvrtc() {
        assert!(check_driver_reads((13, 1), (13, 1)).is_ok());
        assert!(check_driver_reads((13, 1), (13, 0)).is_ok());
        assert!(check_driver_reads((13, 0), (12, 9)).is_ok());
        let err = check_driver_reads((13, 0), (13, 1))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "the NVIDIA driver supports CUDA 13.0, older than NVRTC 13.1: update the driver"
        );
        assert!(check_driver_reads((12, 9), (13, 0)).is_err());
    }

    /// A load that finds nothing says so by its kind, not by its message.
    #[test]
    fn not_found_is_its_own_failure() {
        let failure = load(Platform::Windows, &[], "cublas64_13.dll", |_| Vec::new())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            failure,
            LoadFailure::NotFound("cublas64_13.dll".to_string())
        );
        assert_eq!(failure.text(), "cublas64_13.dll was not found");
    }

    /// The info's text names the card, each version and each file.
    #[test]
    fn info_prints_versions_and_files() {
        let info = CudaInfo {
            name: "NVIDIA GeForce GTX 1660 SUPER".to_string(),
            memory: 6 << 30,
            compute_capability: (7, 5),
            driver_cuda: (13, 1),
            cudnn: (9, 24, 0),
            cudnn_cuda: (13, 3),
            cublas: (13, 0, 0),
            nvrtc: (13, 0),
            compiles_for: (7, 5),
            libraries: vec![PathBuf::from("cudnn64_9.dll")],
        };
        assert_eq!(
            info.to_string(),
            "NVIDIA GeForce GTX 1660 SUPER, 6.0 GiB, compute capability 7.5\n\
             driver supports CUDA 13.1; cuDNN 9.24.0 (built for CUDA 13.3), cuBLAS 13.0.0, \
             NVRTC 13.0 (compiling for 7.5)\n\
             loaded: cudnn64_9.dll"
        );
    }
}
