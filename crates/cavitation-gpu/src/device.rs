//! The device: choosing a GPU adapter among those wgpu offers, and opening its device and
//! queue with the features and limits the kernels rely on.

use std::fmt;
use std::str::FromStr;

/// The environment variable that restricts every [`Gpu`] to one backend.
const BACKEND_VAR: &str = "CAVITATION_GPU_BACKEND";

/// The environment variable that, set to `1`, makes a missing GPU a panic.
const REQUIRED_VAR: &str = "CAVITATION_GPU_REQUIRED";

/// The workgroup memory requested, in bytes: 32 KiB, all that DX12 offers (Vulkan and Metal
/// often have more), or the adapter's if it has less. A kernel's tiles must fit in it.
const WORKGROUP_STORAGE: u32 = 32 * 1024;

/// Parts of adapter names that mark a software renderer or a paravirtual device, in lower
/// case: such adapters are never chosen, whatever type they report.
const NOT_HARDWARE: [&str; 7] = [
    "microsoft basic render driver",
    "warp",
    "llvmpipe",
    "lavapipe",
    "softpipe",
    "swiftshader",
    "paravirtual",
];

/// A GPU backend that [`Gpu`] can run on.
///
/// The OpenGL backend that wgpu also has is not one of them: it lacks parts of compute
/// that the kernels need, and every platform with an OpenGL driver for its GPU has one of
/// these three as well.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::Backend;
///
/// let backend: Backend = "DX12".parse().unwrap();
/// assert_eq!(backend, Backend::Dx12);
/// assert_eq!(backend.to_string(), "dx12");
/// assert!("opengl".parse::<Backend>().is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Vulkan: Windows and Linux, the default on both.
    Vulkan,
    /// Direct3D 12: Windows.
    Dx12,
    /// Metal: macOS, the default there.
    Metal,
}

impl Backend {
    /// Every backend, in the order their names are listed in messages.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Backend;
    ///
    /// let names: Vec<String> = Backend::ALL.iter().map(|b| b.to_string()).collect();
    /// assert_eq!(names, ["vulkan", "dx12", "metal"]);
    /// ```
    pub const ALL: [Backend; 3] = [Backend::Vulkan, Backend::Dx12, Backend::Metal];

    /// The backend's name as `CAVITATION_GPU_BACKEND` and [`FromStr`] take it: `vulkan`,
    /// `dx12` or `metal`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Backend;
    ///
    /// assert_eq!(Backend::Metal.name(), "metal");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Dx12 => "dx12",
            Backend::Metal => "metal",
        }
    }

    /// The same backend in wgpu's terms.
    fn to_wgpu(self) -> wgpu::Backend {
        match self {
            Backend::Vulkan => wgpu::Backend::Vulkan,
            Backend::Dx12 => wgpu::Backend::Dx12,
            Backend::Metal => wgpu::Backend::Metal,
        }
    }

    /// The backend for one of wgpu's, or `None` for OpenGL, the browser's WebGPU and the
    /// no-op backend, which [`Gpu`] does not use.
    fn from_wgpu(backend: wgpu::Backend) -> Option<Backend> {
        match backend {
            wgpu::Backend::Vulkan => Some(Backend::Vulkan),
            wgpu::Backend::Dx12 => Some(Backend::Dx12),
            wgpu::Backend::Metal => Some(Backend::Metal),
            _ => None,
        }
    }

    /// The backends [`Gpu::new`] tries, in order, on the platform this was built for:
    /// Metal on macOS; Vulkan, then DX12 on Windows; Vulkan elsewhere. This ignores
    /// `CAVITATION_GPU_BACKEND`, which [`Gpu::new`] and [`Gpu::with_backend`] apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Backend;
    ///
    /// let backends = Backend::on_this_platform();
    /// assert!(!backends.is_empty());
    /// if cfg!(target_os = "macos") {
    ///     assert_eq!(backends, [Backend::Metal]);
    /// }
    /// ```
    pub fn on_this_platform() -> &'static [Backend] {
        if cfg!(target_vendor = "apple") {
            &[Backend::Metal]
        } else if cfg!(windows) {
            &[Backend::Vulkan, Backend::Dx12]
        } else {
            &[Backend::Vulkan]
        }
    }
}

impl fmt::Display for Backend {
    /// Writes [`Backend::name`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Backend {
    /// [`GpuError::UnknownBackend`], holding the text that was not a backend name.
    type Err = GpuError;

    /// Reads a backend name, ignoring case and surrounding spaces: `vulkan`, `dx12` or
    /// `metal`.
    fn from_str(s: &str) -> Result<Backend, GpuError> {
        let wanted = s.trim().to_ascii_lowercase();
        Backend::ALL
            .into_iter()
            .find(|b| b.name() == wanted)
            .ok_or_else(|| GpuError::UnknownBackend(s.to_string()))
    }
}

/// Why no [`Gpu`] could be opened.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Backend, GpuError};
///
/// let err = "cuda".parse::<Backend>().unwrap_err();
/// assert!(matches!(err, GpuError::UnknownBackend(_)));
/// assert!(err.to_string().contains("vulkan, dx12 or metal"));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// A backend name, given in the program or in `CAVITATION_GPU_BACKEND`, is not one of
    /// `vulkan`, `dx12` or `metal`.
    UnknownBackend(String),
    /// No hardware adapter was found on the backends that were tried.
    NoAdapter {
        /// The backends that were tried, after `CAVITATION_GPU_BACKEND`'s restriction.
        tried: Vec<Backend>,
        /// The backend `CAVITATION_GPU_BACKEND` restricted the search to, if it was set.
        restricted_to: Option<Backend>,
        /// Every adapter wgpu offered on those backends, with the reason it was left out;
        /// also a note when DX12 offered none because DXC was asked for and not found.
        skipped: Vec<String>,
    },
    /// Hardware adapters were found, but none of their devices could be opened.
    Device {
        /// For each adapter tried, in order: its name and backend, and wgpu's message.
        failures: Vec<String>,
    },
}

impl fmt::Display for GpuError {
    /// Writes what went wrong in one sentence, followed by the adapters left out, if any.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::UnknownBackend(name) => {
                write!(
                    f,
                    "unknown GPU backend {name:?}: expected vulkan, dx12 or metal"
                )
            }
            GpuError::NoAdapter {
                tried,
                restricted_to,
                skipped,
            } => {
                let names: Vec<&str> = tried.iter().map(|b| b.name()).collect();
                if names.is_empty() {
                    write!(f, "no GPU backend to try")?;
                } else {
                    write!(f, "no hardware GPU adapter on {}", names.join(", "))?;
                }
                if let Some(backend) = restricted_to {
                    write!(f, " ({BACKEND_VAR} restricts the search to {backend})")?;
                }
                for adapter in skipped {
                    write!(f, "; left out: {adapter}")?;
                }
                Ok(())
            }
            GpuError::Device { failures } => {
                write!(f, "could not open a GPU device: {}", failures.join("; "))
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// What a [`Gpu`] runs on, as [`Gpu::info`] gives it.
///
/// Its [`Display`](fmt::Display) form is a few lines for a log or a benchmark's header.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::Gpu;
///
/// if let Some(gpu) = Gpu::new() {
///     let info = gpu.info();
///     assert!(info.limits.max_compute_workgroup_storage_size <= 32 * 1024);
///     println!("{info}");
/// }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GpuInfo {
    /// The backend in use.
    pub backend: Backend,
    /// The adapter's name, such as "NVIDIA GeForce GTX 1660 SUPER".
    pub name: String,
    /// Whether the adapter is a discrete or an integrated GPU (or of another type).
    pub device_type: wgpu::DeviceType,
    /// The PCI vendor id (0x10de for NVIDIA), or the backend's own id where there is none.
    pub vendor: u32,
    /// The PCI device id, or the backend's own id where there is none.
    pub device: u32,
    /// The driver's name, as the backend reports it (may be empty).
    pub driver: String,
    /// The driver's version or other details, as the backend reports them (may be empty).
    pub driver_info: String,
    /// Whether GPU timestamps are available (`TIMESTAMP_QUERY` was offered and requested).
    pub timestamps: bool,
    /// On DX12, the shader compiler setting in force: `auto` (DXC when `dxcompiler.dll`
    /// loads, FXC otherwise), `dxc` or `fxc`, from `WGPU_DX12_COMPILER`. `None` on other
    /// backends.
    pub dx12_compiler: Option<String>,
    /// The limits the device was opened with: wgpu's defaults, with 32 KiB of workgroup
    /// memory (or the adapter's, if less) and the adapter's own buffer and binding sizes.
    pub limits: wgpu::Limits,
}

impl fmt::Display for GpuInfo {
    /// Writes the adapter, its driver, the timestamp support and the limits the kernels
    /// depend on, one item per line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "gpu: {} ({:?}, {})",
            self.name, self.device_type, self.backend
        )?;
        writeln!(
            f,
            "gpu vendor 0x{:04x}, device 0x{:04x}",
            self.vendor, self.device
        )?;
        writeln!(f, "gpu driver: {} {}", self.driver, self.driver_info)?;
        if let Some(compiler) = &self.dx12_compiler {
            writeln!(f, "gpu dx12 shader compiler: {compiler}")?;
        }
        writeln!(
            f,
            "gpu timestamps: {}",
            if self.timestamps { "yes" } else { "no" }
        )?;
        let l = &self.limits;
        write!(
            f,
            "gpu limits: workgroup memory {} B, storage binding {} B, buffer {} B, {} invocations per workgroup, {} workgroups per dimension",
            l.max_compute_workgroup_storage_size,
            l.max_storage_buffer_binding_size,
            l.max_buffer_size,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroups_per_dimension,
        )
    }
}

/// One GPU: a hardware adapter with its device and queue, open and ready for compute.
///
/// [`Gpu::new`] chooses the adapter; [`Gpu::with_backend`] chooses one on a given backend.
/// Among several adapters on a backend, a discrete GPU is preferred to an integrated one,
/// and the first one wgpu lists wins a tie (wgpu can list the same GPU twice). If a device
/// fails to open, the next adapter in that order is tried, then the next backend. Software
/// renderers (such as Microsoft's Basic Render Driver, llvmpipe and SwiftShader), adapters
/// of type CPU or virtual GPU and paravirtual devices are never chosen.
///
/// The device is opened with `TIMESTAMP_QUERY` when the adapter offers it, and with wgpu's
/// default limits except three: 32 KiB of workgroup memory (or the adapter's, if less), and
/// the adapter's own largest storage binding and buffer. Nothing else is asked for: the
/// kernels use only what every backend has, so a result does not depend on where it runs
/// beyond the error bounds each kernel states.
///
/// The instance takes wgpu's own environment variables too, such as `WGPU_VALIDATION` and
/// `WGPU_DEBUG`. A clone shares the same device and queue.
///
/// # Examples
///
/// ```
/// use cavitation_gpu::{Backend, Gpu};
///
/// // Each is `None` without a suitable GPU.
/// if let Some(gpu) = Gpu::new() {
///     println!("{}", gpu.info());
/// }
/// if let Some(gpu) = Gpu::with_backend(Backend::Vulkan) {
///     assert_eq!(gpu.info().backend, Backend::Vulkan);
/// }
/// ```
#[derive(Clone)]
pub struct Gpu {
    /// The adapter the device was opened on.
    adapter: wgpu::Adapter,
    /// The open device.
    device: wgpu::Device,
    /// The device's queue.
    queue: wgpu::Queue,
    /// What was chosen, for [`Gpu::info`].
    info: GpuInfo,
}

impl fmt::Debug for Gpu {
    /// Writes the [`GpuInfo`]; the wgpu handles have nothing more to show.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl Gpu {
    /// Opens the best hardware GPU on this platform's backends, or returns `None` if there
    /// is none.
    ///
    /// The backends are tried in order: Metal on macOS; Vulkan, then DX12 on Windows;
    /// Vulkan elsewhere. `CAVITATION_GPU_BACKEND` replaces that list with the one backend
    /// it names. [`Gpu::open`] gives the reason when there is no GPU.
    ///
    /// # Panics
    ///
    /// If `CAVITATION_GPU_BACKEND` is set to something other than `vulkan`, `dx12` or
    /// `metal`, so that a misspelt name does not quietly skip every GPU test; and if no GPU
    /// can be opened while `CAVITATION_GPU_REQUIRED` is `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Gpu;
    ///
    /// match Gpu::new() {
    ///     Some(gpu) => println!("{}", gpu.info()),
    ///     None => println!("no GPU: running on the CPU"),
    /// }
    /// ```
    pub fn new() -> Option<Gpu> {
        Gpu::or_skip(
            Gpu::open(Backend::on_this_platform()),
            true,
            required_from_env(),
        )
    }

    /// Opens the best hardware GPU on one backend, or returns `None` if there is none
    /// there, or if `CAVITATION_GPU_BACKEND` names another backend.
    ///
    /// # Panics
    ///
    /// If `CAVITATION_GPU_BACKEND` is not a backend name; and if no GPU can be opened while
    /// `CAVITATION_GPU_REQUIRED` is `1`, the backend is one [`Gpu::new`] tries on this
    /// platform, and `CAVITATION_GPU_BACKEND` does not name another. So a loop over
    /// [`Backend::ALL`] skips Metal on Windows, and the backends the restriction excludes,
    /// even when a GPU is required.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Backend, Gpu};
    ///
    /// for backend in Backend::ALL {
    ///     if let Some(gpu) = Gpu::with_backend(backend) {
    ///         assert_eq!(gpu.info().backend, backend);
    ///     }
    /// }
    /// ```
    pub fn with_backend(backend: Backend) -> Option<Gpu> {
        let result = Gpu::open(&[backend]);
        let expected =
            restriction_from_env().is_ok_and(|restricted_to| expected_here(backend, restricted_to));
        Gpu::or_skip(result, expected, required_from_env())
    }

    /// Opens the best hardware GPU on the first of `backends` that has one, after
    /// `CAVITATION_GPU_BACKEND`'s restriction, and says why when none can be opened.
    ///
    /// Unlike [`Gpu::new`] and [`Gpu::with_backend`], this never panics and ignores
    /// `CAVITATION_GPU_REQUIRED`: an unknown `CAVITATION_GPU_BACKEND` is
    /// [`GpuError::UnknownBackend`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Backend, Gpu, GpuError};
    ///
    /// match Gpu::open(&[Backend::Metal, Backend::Vulkan]) {
    ///     Ok(gpu) => println!("{}", gpu.info()),
    ///     Err(GpuError::NoAdapter { skipped, .. }) => println!("no GPU; left out: {skipped:?}"),
    ///     Err(err) => println!("{err}"),
    /// }
    /// ```
    pub fn open(backends: &[Backend]) -> Result<Gpu, GpuError> {
        let restricted_to = restriction_from_env()?;
        let tried = allowed(backends, restricted_to);
        let mut wanted = wgpu::Backends::empty();
        for backend in &tried {
            wanted |= wgpu::Backends::from(backend.to_wgpu());
        }
        let compiler = wgpu::Dx12Compiler::Auto.with_env();
        let dx12_compiler = describe_dx12_compiler(&compiler);
        let dynamic_dxc = matches!(compiler, wgpu::Dx12Compiler::DynamicDxc { .. });
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wanted;
        descriptor.flags = wgpu::InstanceFlags::from_build_config().with_env();
        descriptor.backend_options.dx12.shader_compiler = compiler;
        let instance = wgpu::Instance::new(descriptor);
        let adapters = if tried.is_empty() {
            Vec::new()
        } else {
            pollster::block_on(instance.enumerate_adapters(wanted))
        };
        let candidates: Vec<Candidate> = adapters
            .iter()
            .map(|a| Candidate::from(&a.get_info()))
            .collect();
        let order = ranked(&candidates, &tried);
        if order.is_empty() {
            let mut skipped: Vec<String> = candidates
                .iter()
                .filter_map(|c| {
                    c.left_out_because().map(|why| {
                        format!("{} ({:?}, {:?}): {why}", c.name, c.backend, c.device_type)
                    })
                })
                .collect();
            let dx12_listed = candidates.iter().any(|c| c.backend == wgpu::Backend::Dx12);
            if tried.contains(&Backend::Dx12) && !dx12_listed && dynamic_dxc {
                skipped.push(
                    "every DX12 adapter: WGPU_DX12_COMPILER asks for DXC, and dxcompiler.dll \
                     could not be loaded (it must be on PATH or beside the program)"
                        .to_string(),
                );
            }
            return Err(GpuError::NoAdapter {
                tried,
                restricted_to,
                skipped,
            });
        }
        let mut failures = Vec::new();
        for index in order {
            match open_device(&adapters[index], &dx12_compiler) {
                Ok(gpu) => return Ok(gpu),
                Err(failure) => failures.push(failure),
            }
        }
        Err(GpuError::Device { failures })
    }

    /// What this GPU is: its backend, adapter, driver and the limits it was opened with.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Gpu;
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     println!("running on {}", gpu.info().name);
    /// }
    /// ```
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// The wgpu adapter the device was opened on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Gpu;
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     assert_eq!(gpu.adapter().get_info().name, gpu.info().name);
    /// }
    /// ```
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// The wgpu device, for creating buffers and pipelines.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::{Gpu, wgpu};
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
    ///         label: None,
    ///         size: 16,
    ///         usage: wgpu::BufferUsages::STORAGE,
    ///         mapped_at_creation: false,
    ///     });
    ///     assert_eq!(buffer.size(), 16);
    /// }
    /// ```
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The device's queue, for writing buffers and submitting work.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_gpu::Gpu;
    ///
    /// if let Some(gpu) = Gpu::new() {
    ///     // An empty submission completes at once.
    ///     gpu.queue().submit([]);
    /// }
    /// ```
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Turns [`Gpu::open`]'s result into `Some` or `None`. It panics on an unknown backend
    /// name, and on any other error when a GPU is `expected` there and `required`
    /// (`CAVITATION_GPU_REQUIRED` is `1`).
    fn or_skip(result: Result<Gpu, GpuError>, expected: bool, required: bool) -> Option<Gpu> {
        match result {
            Ok(gpu) => Some(gpu),
            Err(err @ GpuError::UnknownBackend(_)) => panic!("{BACKEND_VAR}: {err}"),
            Err(err) if expected && required => panic!("{REQUIRED_VAR} is 1, but {err}"),
            Err(_) => None,
        }
    }
}

/// Opens the device of one eligible adapter, or says why it failed: its name and backend
/// with wgpu's message.
fn open_device(adapter: &wgpu::Adapter, dx12_compiler: &str) -> Result<Gpu, String> {
    let adapter_info = adapter.get_info();
    let backend = Backend::from_wgpu(adapter_info.backend)
        .expect("only Vulkan, DX12 and Metal adapters are eligible");
    let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let limits = requested_limits(&adapter.limits());
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("cavitation"),
        required_features: if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        required_limits: limits.clone(),
        ..Default::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
        .map_err(|e| format!("{} ({backend}): {e}", adapter_info.name))?;
    let info = GpuInfo {
        backend,
        name: adapter_info.name,
        device_type: adapter_info.device_type,
        vendor: adapter_info.vendor,
        device: adapter_info.device,
        driver: adapter_info.driver,
        driver_info: adapter_info.driver_info,
        timestamps,
        dx12_compiler: (backend == Backend::Dx12).then(|| dx12_compiler.to_string()),
        limits,
    };
    Ok(Gpu {
        adapter: adapter.clone(),
        device,
        queue,
        info,
    })
}

/// What [`choose`] needs to know about one adapter wgpu offers.
#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    /// The adapter's backend.
    backend: wgpu::Backend,
    /// Its device type.
    device_type: wgpu::DeviceType,
    /// Its name.
    name: String,
    /// Its vendor id.
    vendor: u32,
    /// Its device id.
    device: u32,
}

impl From<&wgpu::AdapterInfo> for Candidate {
    /// Keeps the fields of the adapter's information that [`choose`] looks at.
    fn from(info: &wgpu::AdapterInfo) -> Candidate {
        Candidate {
            backend: info.backend,
            device_type: info.device_type,
            name: info.name.clone(),
            vendor: info.vendor,
            device: info.device,
        }
    }
}

impl Candidate {
    /// Why this adapter is never chosen, or `None` if it is a hardware GPU on a backend
    /// [`Gpu`] uses.
    fn left_out_because(&self) -> Option<&'static str> {
        if Backend::from_wgpu(self.backend).is_none() {
            return Some("backend not used");
        }
        if matches!(
            self.device_type,
            wgpu::DeviceType::Cpu | wgpu::DeviceType::VirtualGpu
        ) {
            return Some("not a hardware GPU");
        }
        let name = self.name.to_ascii_lowercase();
        if NOT_HARDWARE.iter().any(|marker| name.contains(marker)) {
            return Some("software or paravirtual");
        }
        None
    }

    /// The order in which device types are preferred: discrete, integrated, then other.
    fn rank(&self) -> u8 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        }
    }
}

/// The indices of the eligible adapters in the order to try them: backend by backend as in
/// `order`, and on each backend by device type (discrete first), then as listed. A GPU
/// listed twice on one backend (same vendor and device id) appears once, at its first
/// listing. Empty when no backend in `order` has an eligible adapter.
fn ranked(candidates: &[Candidate], order: &[Backend]) -> Vec<usize> {
    let mut indices = Vec::new();
    for backend in order {
        let mut on_backend: Vec<(usize, &Candidate)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.backend == backend.to_wgpu() && c.left_out_because().is_none())
            .collect();
        on_backend.sort_by_key(|(i, c)| (c.rank(), *i));
        let mut seen: Vec<(u32, u32)> = Vec::new();
        for (i, c) in on_backend {
            if !seen.contains(&(c.vendor, c.device)) {
                seen.push((c.vendor, c.device));
                indices.push(i);
            }
        }
    }
    indices
}

/// The backends to try: `order`, or only the restricting backend when one is set (nothing
/// if `order` does not contain it).
fn allowed(order: &[Backend], restricted_to: Option<Backend>) -> Vec<Backend> {
    let mut tried: Vec<Backend> = Vec::new();
    for &backend in order {
        if restricted_to.is_none_or(|r| r == backend) && !tried.contains(&backend) {
            tried.push(backend);
        }
    }
    tried
}

/// Whether a GPU on `backend` should be there, so that `CAVITATION_GPU_REQUIRED` applies
/// to it: the platform's [`Gpu::new`] tries it, and the restriction, if any, names it.
fn expected_here(backend: Backend, restricted_to: Option<Backend>) -> bool {
    Backend::on_this_platform().contains(&backend) && restricted_to.is_none_or(|r| r == backend)
}

/// Reads the restriction from `CAVITATION_GPU_BACKEND`; a value that is not valid Unicode
/// is an unknown backend name, as a misspelt one is.
fn restriction_from_env() -> Result<Option<Backend>, GpuError> {
    match std::env::var(BACKEND_VAR) {
        Ok(value) => parse_restriction(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(value)) => Err(GpuError::UnknownBackend(
            value.to_string_lossy().into_owned(),
        )),
    }
}

/// Whether `CAVITATION_GPU_REQUIRED` makes a GPU required.
fn required_from_env() -> bool {
    parse_required(std::env::var(REQUIRED_VAR).ok().as_deref())
}

/// Reads `CAVITATION_GPU_BACKEND`'s value: unset or empty means no restriction.
fn parse_restriction(value: Option<&str>) -> Result<Option<Backend>, GpuError> {
    match value {
        None => Ok(None),
        Some(v) if v.trim().is_empty() => Ok(None),
        Some(v) => v.parse().map(Some),
    }
}

/// Reads `CAVITATION_GPU_REQUIRED`'s value: only `1` (with surrounding spaces allowed)
/// makes a GPU required.
fn parse_required(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// The limits to open the device with, given the adapter's: wgpu's defaults, except the
/// workgroup memory (32 KiB, or the adapter's if less) and the largest storage binding and
/// buffer (the adapter's).
fn requested_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_compute_workgroup_storage_size: WORKGROUP_STORAGE
            .min(adapter.max_compute_workgroup_storage_size),
        max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
        max_buffer_size: adapter.max_buffer_size,
        ..wgpu::Limits::default()
    }
}

/// The DX12 shader compiler setting in words, for [`GpuInfo::dx12_compiler`].
fn describe_dx12_compiler(compiler: &wgpu::Dx12Compiler) -> String {
    match compiler {
        wgpu::Dx12Compiler::Auto => "auto (DXC if dxcompiler.dll loads, FXC otherwise)".to_string(),
        wgpu::Dx12Compiler::DynamicDxc { dxc_path } => format!("dxc ({dxc_path})"),
        wgpu::Dx12Compiler::StaticDxc => "dxc (linked in)".to_string(),
        wgpu::Dx12Compiler::Fxc => "fxc".to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! The backend names, the adapter choice on made-up adapter lists, the environment
    //! variables' parsing and the requested limits; none of these needs a GPU.

    use super::*;

    /// A made-up adapter description.
    fn adapter(
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        name: &str,
        vendor: u32,
        device: u32,
    ) -> Candidate {
        Candidate {
            backend,
            device_type,
            name: name.to_string(),
            vendor,
            device,
        }
    }

    /// The adapters wgpu 30 lists on a Windows desktop with an NVIDIA GTX 1660 Super: the
    /// card on Vulkan, twice on DX12, Microsoft's software renderer and the card on OpenGL.
    fn windows_desktop() -> Vec<Candidate> {
        use wgpu::{Backend as B, DeviceType as T};
        vec![
            adapter(
                B::Vulkan,
                T::DiscreteGpu,
                "NVIDIA GeForce GTX 1660 SUPER",
                0x10de,
                0x21c4,
            ),
            adapter(
                B::Dx12,
                T::DiscreteGpu,
                "NVIDIA GeForce GTX 1660 SUPER",
                0x10de,
                0x21c4,
            ),
            adapter(
                B::Dx12,
                T::DiscreteGpu,
                "NVIDIA GeForce GTX 1660 SUPER",
                0x10de,
                0x21c4,
            ),
            adapter(
                B::Dx12,
                T::Cpu,
                "Microsoft Basic Render Driver",
                0x1414,
                0x008c,
            ),
            adapter(
                B::Gl,
                T::Other,
                "NVIDIA GeForce GTX 1660 SUPER/PCIe/SSE2",
                0x10de,
                0,
            ),
        ]
    }

    /// The adapter [`Gpu::open`] tries first.
    fn choose(candidates: &[Candidate], order: &[Backend]) -> Option<usize> {
        ranked(candidates, order).first().copied()
    }

    /// The whole order on the Windows desktop: the card on Vulkan, then its first DX12
    /// listing; the second listing, the software renderer and OpenGL are not tried.
    #[test]
    fn ranks_the_windows_desktop() {
        let list = windows_desktop();
        assert_eq!(ranked(&list, &[Backend::Vulkan, Backend::Dx12]), vec![0, 1]);
        assert_eq!(ranked(&list, &[Backend::Dx12, Backend::Vulkan]), vec![1, 0]);
        assert_eq!(ranked(&list, &Backend::ALL), vec![0, 1]);
    }

    /// The backend order comes before the device type: an integrated GPU on the first
    /// backend is tried before a discrete one on the next, and both are tried.
    #[test]
    fn backend_order_beats_device_type() {
        use wgpu::{Backend as B, DeviceType as T};
        let list = vec![
            adapter(
                B::Dx12,
                T::DiscreteGpu,
                "NVIDIA GeForce RTX 3060",
                0x10de,
                0x2503,
            ),
            adapter(
                B::Vulkan,
                T::IntegratedGpu,
                "AMD Radeon(TM) Graphics",
                0x1002,
                0x1638,
            ),
        ];
        assert_eq!(ranked(&list, &[Backend::Vulkan, Backend::Dx12]), vec![1, 0]);
    }

    /// Two different adapters of type Other on one backend are both tried, in the order
    /// they are listed.
    #[test]
    fn adapters_of_type_other_keep_their_order() {
        use wgpu::{Backend as B, DeviceType as T};
        let list = vec![
            adapter(B::Vulkan, T::Other, "GPU A", 0x1234, 1),
            adapter(B::Vulkan, T::Other, "GPU B", 0x1234, 2),
        ];
        assert_eq!(ranked(&list, &[Backend::Vulkan]), vec![0, 1]);
    }

    /// The DX12 compiler settings read as the names `WGPU_DX12_COMPILER` takes.
    #[test]
    fn dx12_compiler_descriptions() {
        assert!(describe_dx12_compiler(&wgpu::Dx12Compiler::Auto).starts_with("auto"));
        assert_eq!(describe_dx12_compiler(&wgpu::Dx12Compiler::Fxc), "fxc");
        assert_eq!(
            describe_dx12_compiler(&wgpu::Dx12Compiler::default_dynamic_dxc()),
            "dxc (dxcompiler.dll)"
        );
        assert_eq!(
            describe_dx12_compiler(&wgpu::Dx12Compiler::StaticDxc),
            "dxc (linked in)"
        );
    }

    /// Without a GPU, the result is `None` unless a GPU is both expected and required.
    #[test]
    fn missing_gpu_is_none_unless_expected_and_required() {
        let missing = || GpuError::NoAdapter {
            tried: vec![Backend::Vulkan],
            restricted_to: None,
            skipped: Vec::new(),
        };
        assert!(Gpu::or_skip(Err(missing()), false, false).is_none());
        assert!(Gpu::or_skip(Err(missing()), true, false).is_none());
        assert!(Gpu::or_skip(Err(missing()), false, true).is_none());
        let failed = GpuError::Device {
            failures: vec!["GPU (vulkan): out of memory".to_string()],
        };
        assert!(Gpu::or_skip(Err(failed), true, false).is_none());
    }

    /// A required GPU that should be there and is missing is a panic naming the variable.
    #[test]
    #[should_panic(
        expected = "CAVITATION_GPU_REQUIRED is 1, but no hardware GPU adapter on vulkan"
    )]
    fn missing_required_gpu_panics() {
        let missing = GpuError::NoAdapter {
            tried: vec![Backend::Vulkan],
            restricted_to: None,
            skipped: Vec::new(),
        };
        Gpu::or_skip(Err(missing), true, true);
    }

    /// A device that would not open, when required, is a panic with wgpu's message.
    #[test]
    #[should_panic(expected = "could not open a GPU device: GPU (vulkan): out of memory")]
    fn failed_required_device_panics() {
        let failed = GpuError::Device {
            failures: vec!["GPU (vulkan): out of memory".to_string()],
        };
        Gpu::or_skip(Err(failed), true, true);
    }

    /// A misspelt backend name is a panic whether or not a GPU is required.
    #[test]
    #[should_panic(expected = "CAVITATION_GPU_BACKEND: unknown GPU backend \"vulcan\"")]
    fn unknown_backend_always_panics() {
        Gpu::or_skip(
            Err(GpuError::UnknownBackend("vulcan".to_string())),
            false,
            false,
        );
    }

    /// With no backend left to try, the message says so rather than naming none.
    #[test]
    fn no_backend_message() {
        let err = GpuError::NoAdapter {
            tried: Vec::new(),
            restricted_to: Some(Backend::Metal),
            skipped: Vec::new(),
        };
        assert!(
            err.to_string().starts_with("no GPU backend to try"),
            "{err}"
        );
    }

    /// Backend names parse in any case with spaces around them, and print back as their
    /// lower-case names.
    #[test]
    fn backend_names_parse_and_print() {
        for (text, backend) in [
            ("vulkan", Backend::Vulkan),
            (" Vulkan ", Backend::Vulkan),
            ("DX12", Backend::Dx12),
            ("metal", Backend::Metal),
        ] {
            assert_eq!(text.parse::<Backend>().unwrap(), backend, "{text:?}");
        }
        for backend in Backend::ALL {
            assert_eq!(backend.to_string().parse::<Backend>().unwrap(), backend);
        }
    }

    /// Anything else is an error that names the input and the accepted names.
    #[test]
    fn unknown_backend_names_are_errors() {
        for text in ["", "gl", "opengl", "d3d12", "cuda", "vulkan2"] {
            let err = text.parse::<Backend>().unwrap_err();
            assert!(
                matches!(&err, GpuError::UnknownBackend(t) if t == text),
                "{text:?}"
            );
            assert!(
                err.to_string().contains("expected vulkan, dx12 or metal"),
                "{err}"
            );
        }
    }

    /// On the Windows desktop, Vulkan comes first; asked for DX12, the first of the two
    /// identical listings is chosen; Metal has nothing.
    #[test]
    fn chooses_on_the_windows_desktop() {
        let list = windows_desktop();
        assert_eq!(choose(&list, &[Backend::Vulkan, Backend::Dx12]), Some(0));
        assert_eq!(choose(&list, &[Backend::Dx12, Backend::Vulkan]), Some(1));
        assert_eq!(choose(&list, &[Backend::Dx12]), Some(1));
        assert_eq!(choose(&list, &[Backend::Metal]), None);
    }

    /// A later backend in the order is used when the first has no eligible adapter.
    #[test]
    fn falls_back_to_the_next_backend() {
        let list: Vec<Candidate> = windows_desktop()
            .into_iter()
            .filter(|c| c.backend != wgpu::Backend::Vulkan)
            .collect();
        assert_eq!(choose(&list, &[Backend::Vulkan, Backend::Dx12]), Some(0));
    }

    /// Software renderers, CPU and virtual adapters and paravirtual devices are never
    /// chosen, whatever their type says, and OpenGL is never used.
    #[test]
    fn leaves_out_software_paravirtual_and_opengl() {
        use wgpu::{Backend as B, DeviceType as T};
        let list = vec![
            adapter(
                B::Dx12,
                T::Cpu,
                "Microsoft Basic Render Driver",
                0x1414,
                0x8c,
            ),
            adapter(
                B::Dx12,
                T::IntegratedGpu,
                "Microsoft Basic Render Driver",
                0x1414,
                0x8c,
            ),
            adapter(
                B::Vulkan,
                T::Cpu,
                "llvmpipe (LLVM 19.1.7, 256 bits)",
                0x10005,
                0,
            ),
            adapter(
                B::Vulkan,
                T::Other,
                "SwiftShader Device (Subzero)",
                0x1ae0,
                0xc0de,
            ),
            adapter(
                B::Dx12,
                T::IntegratedGpu,
                "Microsoft WARP Adapter",
                0x1414,
                0x8c,
            ),
            adapter(B::Metal, T::Other, "Apple Paravirtual device", 0, 0),
            adapter(B::Metal, T::VirtualGpu, "Some virtual GPU", 0, 0),
            adapter(
                B::Gl,
                T::DiscreteGpu,
                "NVIDIA GeForce GTX 1660 SUPER/PCIe/SSE2",
                0x10de,
                0,
            ),
        ];
        assert_eq!(choose(&list, &Backend::ALL), None);
        for c in &list {
            assert!(c.left_out_because().is_some(), "{c:?}");
        }
    }

    /// A discrete GPU beats an integrated one on the same backend, even when listed later.
    #[test]
    fn prefers_a_discrete_gpu() {
        use wgpu::{Backend as B, DeviceType as T};
        let list = vec![
            adapter(
                B::Vulkan,
                T::IntegratedGpu,
                "Intel(R) UHD Graphics 630",
                0x8086,
                0x3e92,
            ),
            adapter(
                B::Vulkan,
                T::DiscreteGpu,
                "NVIDIA GeForce RTX 3060",
                0x10de,
                0x2503,
            ),
        ];
        assert_eq!(choose(&list, &[Backend::Vulkan]), Some(1));
        let apple = vec![adapter(B::Metal, T::IntegratedGpu, "Apple M4", 0, 0)];
        assert_eq!(choose(&apple, &[Backend::Metal]), Some(0));
    }

    /// `CAVITATION_GPU_BACKEND` narrows the backends tried to the one it names, or to
    /// nothing if that one was not asked for.
    #[test]
    fn restriction_narrows_the_backends() {
        let order = [Backend::Vulkan, Backend::Dx12];
        assert_eq!(allowed(&order, None), order.to_vec());
        assert_eq!(allowed(&order, Some(Backend::Dx12)), vec![Backend::Dx12]);
        assert_eq!(allowed(&order, Some(Backend::Metal)), Vec::<Backend>::new());
        assert_eq!(
            allowed(&[Backend::Vulkan, Backend::Vulkan], None),
            vec![Backend::Vulkan]
        );
    }

    /// A GPU is expected on the platform's own backends unless the restriction names
    /// another, and never on a backend the platform does not try.
    #[test]
    fn expected_backends() {
        let first = Backend::on_this_platform()[0];
        assert!(expected_here(first, None));
        assert!(expected_here(first, Some(first)));
        for other in Backend::ALL.into_iter().filter(|b| *b != first) {
            assert!(!expected_here(first, Some(other)), "{other}");
            assert_eq!(
                expected_here(other, None),
                Backend::on_this_platform().contains(&other),
                "{other}"
            );
        }
        let missing: Vec<Backend> = Backend::ALL
            .into_iter()
            .filter(|b| !Backend::on_this_platform().contains(b))
            .collect();
        for backend in missing {
            assert!(!expected_here(backend, None), "{backend}");
        }
    }

    /// The environment variables' values: an unset or blank backend means no restriction,
    /// a misspelt one is an error, and only `1` makes a GPU required.
    #[test]
    fn environment_values() {
        assert_eq!(parse_restriction(None).unwrap(), None);
        assert_eq!(parse_restriction(Some("  ")).unwrap(), None);
        assert_eq!(
            parse_restriction(Some("dx12")).unwrap(),
            Some(Backend::Dx12)
        );
        assert!(matches!(
            parse_restriction(Some("vulcan")),
            Err(GpuError::UnknownBackend(_))
        ));
        assert!(parse_required(Some("1")));
        assert!(parse_required(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("yes"), Some("true")] {
            assert!(!parse_required(v), "{v:?}");
        }
    }

    /// The device asks for 32 KiB of workgroup memory (less if the adapter has less) and
    /// the adapter's buffer and binding sizes, and keeps wgpu's defaults elsewhere.
    #[test]
    fn requested_limits_follow_the_adapter() {
        let vulkan = wgpu::Limits {
            max_compute_workgroup_storage_size: 49152,
            max_storage_buffer_binding_size: 2_147_483_644,
            max_buffer_size: 4_292_870_144,
            ..wgpu::Limits::default()
        };
        let limits = requested_limits(&vulkan);
        assert_eq!(limits.max_compute_workgroup_storage_size, 32768);
        assert_eq!(limits.max_storage_buffer_binding_size, 2_147_483_644);
        assert_eq!(limits.max_buffer_size, 4_292_870_144);
        let defaults = wgpu::Limits::default();
        assert_eq!(
            limits.max_storage_buffers_per_shader_stage,
            defaults.max_storage_buffers_per_shader_stage
        );
        assert_eq!(limits.max_bind_groups, defaults.max_bind_groups);
        assert_eq!(
            limits.max_compute_workgroups_per_dimension,
            defaults.max_compute_workgroups_per_dimension
        );
        let small = wgpu::Limits::default();
        assert_eq!(
            requested_limits(&small).max_compute_workgroup_storage_size,
            small.max_compute_workgroup_storage_size
        );
    }

    /// A missing adapter's message names the backends tried, the restriction and every
    /// adapter left out.
    #[test]
    fn no_adapter_message() {
        let err = GpuError::NoAdapter {
            tried: vec![Backend::Dx12],
            restricted_to: Some(Backend::Dx12),
            skipped: vec![
                "Microsoft Basic Render Driver (Dx12, Cpu): not a hardware GPU".to_string(),
            ],
        };
        let text = err.to_string();
        assert!(
            text.starts_with("no hardware GPU adapter on dx12"),
            "{text}"
        );
        assert!(
            text.contains("CAVITATION_GPU_BACKEND restricts the search to dx12"),
            "{text}"
        );
        assert!(
            text.contains("left out: Microsoft Basic Render Driver"),
            "{text}"
        );
    }
}
