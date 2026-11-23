//! The devices a run can train on: [`devices`] lists what this machine and this build offer,
//! best first, and marks the one [`Device::Auto`](crate::Device) picks; `CAVITATION_DEVICE`
//! restricts that pick to CUDA, the wgpu GPU or the CPU.

use std::fmt;
use std::str::FromStr;

use cavitation_core::Error;
use cavitation_core::kernels::{level, threads};

/// The environment variable that restricts [`Device::Auto`](crate::Device) to one kind of
/// device: `auto` (the same as leaving it unset), `cuda`, `gpu` or `cpu`.
const DEVICE_VAR: &str = "CAVITATION_DEVICE";

/// A kind of device a run trains on.
///
/// # Examples
///
/// ```
/// use cavitation_nn::DeviceKind;
///
/// let kind: DeviceKind = "GPU".parse().unwrap();
/// assert_eq!(kind, DeviceKind::Gpu);
/// assert_eq!(kind.to_string(), "gpu");
/// assert!("tpu".parse::<DeviceKind>().is_err());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// The CPU, through [`Trainer`](crate::Trainer).
    #[default]
    Cpu,
    /// A GPU through wgpu (Vulkan, DX12 or Metal), with the `gpu` feature.
    Gpu,
    /// An NVIDIA card through CUDA, cuDNN and cuBLAS, with the `cuda` feature.
    Cuda,
}

impl DeviceKind {
    /// The kind's name as `CAVITATION_DEVICE` and [`FromStr`] take it: `cpu`, `gpu` or
    /// `cuda`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cavitation_nn::DeviceKind;
    ///
    /// assert_eq!(DeviceKind::Cpu.name(), "cpu");
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
            DeviceKind::Cuda => "cuda",
        }
    }
}

impl fmt::Display for DeviceKind {
    /// Writes [`DeviceKind::name`], padded as the format asks.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

impl FromStr for DeviceKind {
    /// [`Error::InvalidData`], naming the text that was not a kind of device.
    type Err = Error;

    /// Reads `cpu`, `gpu` or `cuda`, ignoring case and surrounding spaces.
    fn from_str(s: &str) -> Result<DeviceKind, Error> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(DeviceKind::Cpu),
            "gpu" => Ok(DeviceKind::Gpu),
            "cuda" => Ok(DeviceKind::Cuda),
            _ => Err(Error::InvalidData(format!(
                "unknown device {s:?}: expected cpu, gpu or cuda"
            ))),
        }
    }
}

/// What `CAVITATION_DEVICE` asks of [`Device::Auto`](crate::Device).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Choice {
    /// The best device that can run the job: the variable is unset, empty or `auto`.
    Best,
    /// Only this kind.
    Only(DeviceKind),
}

/// Reads a `CAVITATION_DEVICE` value: unset, empty or `auto` asks for the best device, and
/// `cuda`, `gpu` or `cpu` for that kind only, ignoring case and surrounding spaces.
///
/// # Errors
///
/// [`Error::InvalidData`] for any other value, naming it and the values accepted.
pub(crate) fn parse_choice(value: Option<&str>) -> Result<Choice, Error> {
    let Some(text) = value.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(Choice::Best);
    };
    if text.eq_ignore_ascii_case("auto") {
        return Ok(Choice::Best);
    }
    text.parse().map(Choice::Only).map_err(|_| {
        Error::InvalidData(format!(
            "{DEVICE_VAR}: unknown device {text:?}: expected auto, cuda, gpu or cpu"
        ))
    })
}

/// Reads `CAVITATION_DEVICE` through [`parse_choice`]; a value that is not valid Unicode is
/// an error like any other unknown value.
pub(crate) fn choice_from_env() -> Result<Choice, Error> {
    match std::env::var(DEVICE_VAR) {
        Ok(value) => parse_choice(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_choice(None),
        Err(std::env::VarError::NotUnicode(raw)) => parse_choice(Some(&raw.to_string_lossy())),
    }
}

/// One device in [`devices`]' list.
///
/// Its [`Display`](fmt::Display) form is one line for a command's output: the kind, the
/// name, the detail and the note, or, for a device that is not available, the reason.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{DeviceKind, devices};
///
/// // An error only if `CAVITATION_DEVICE` holds something other than `auto`, `cuda`, `gpu`
/// // or `cpu`.
/// if let Ok(list) = devices() {
///     let cpu = list.last().unwrap();
///     assert_eq!(cpu.kind, DeviceKind::Cpu);
///     assert!(cpu.available);
///     println!("{cpu}");
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceEntry {
    /// The kind of device.
    pub kind: DeviceKind,
    /// The device's name: a card's or a GPU adapter's, such as "NVIDIA GeForce GTX 1660
    /// SUPER", or "CPU". Empty for a device that is not available.
    pub name: String,
    /// How it would run: the CUDA driver's version and the cuDNN and cuBLAS versions
    /// ("CUDA 13.0, cuDNN 9.13.0, cuBLAS 13.1.0"), a GPU's backend, with the platform's other
    /// backends that open the same card ("vulkan (also: dx12)"), or the CPU's kernel level and
    /// thread count ("avx2, 16 threads"). Empty for a device that is not available.
    pub detail: String,
    /// Whether a run can train on it in this build on this machine.
    pub available: bool,
    /// Whether [`Device::Auto`](crate::Device) picks it, after `CAVITATION_DEVICE`. At most
    /// one entry is chosen; none is when the variable asks for a kind that is not available.
    pub chosen: bool,
    /// For a device that is not available, why and what would make it so; for one that is,
    /// anything worth knowing about it, such as a faster engine for the same card.
    pub note: Option<String>,
}

impl fmt::Display for DeviceEntry {
    /// Writes the entry on one line, ending in `<- chosen` for the chosen one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<4}", self.kind.name())?;
        if self.available {
            write!(f, " {}, {}", self.name, self.detail)?;
            if let Some(note) = &self.note {
                write!(f, " ({note})")?;
            }
        } else {
            let why = self.note.as_deref().unwrap_or("no reason given");
            write!(f, " not available: {why}")?;
        }
        if self.chosen {
            write!(f, "  <- chosen")?;
        }
        Ok(())
    }
}

/// The PCI vendor id of NVIDIA, whose cards have a faster engine: CUDA with cuDNN.
#[cfg(feature = "gpu")]
const NVIDIA: u32 = 0x10de;

/// What the wgpu GPU's entry says of an NVIDIA card: that the CUDA engine trains on it
/// faster, given whether CUDA opened (`None` for a build without the `cuda` feature).
#[cfg(feature = "gpu")]
fn nvidia_note(cuda: Option<bool>) -> &'static str {
    match cuda {
        Some(true) => {
            "an NVIDIA card: the cuda entry trains on it faster, through NVIDIA's CUDA and cuDNN"
        }
        Some(false) => {
            "an NVIDIA card: with what the cuda entry names installed, NVIDIA's CUDA and cuDNN \
             would train on it faster"
        }
        None => {
            "an NVIDIA card: a build with the `cuda` feature trains on it faster, through \
             NVIDIA's CUDA and cuDNN"
        }
    }
}

/// CUDA's entry: the card `Cuda::open` opens, with the driver's and the libraries'
/// versions, or why it does not open.
#[cfg(feature = "cuda")]
fn cuda_entry() -> DeviceEntry {
    use cavitation_cuda::Cuda;

    match Cuda::open() {
        Ok(cuda) => {
            let info = cuda.info();
            let (c, d, b) = (info.driver_cuda, info.cudnn, info.cublas);
            DeviceEntry {
                kind: DeviceKind::Cuda,
                name: info.name.clone(),
                detail: format!(
                    "CUDA {}.{}, cuDNN {}.{}.{}, cuBLAS {}.{}.{}",
                    c.0, c.1, d.0, d.1, d.2, b.0, b.1, b.2
                ),
                available: true,
                chosen: false,
                note: None,
            }
        }
        Err(why) => DeviceEntry {
            kind: DeviceKind::Cuda,
            name: String::new(),
            detail: String::new(),
            available: false,
            chosen: false,
            note: Some(why.to_string()),
        },
    }
}

/// The GPU's entry: the adapter `Gpu::open` picks on this platform's backends, or why there
/// is none, with a note on an NVIDIA card given whether CUDA opened (`cuda`, `None` without
/// the `cuda` feature). Opening it is how its presence is known, so this opens a device on
/// each backend of the platform.
#[cfg(feature = "gpu")]
fn gpu_entry(cuda: Option<bool>) -> DeviceEntry {
    use cavitation_gpu::{Backend, Gpu};

    let backends = Backend::on_this_platform();
    match Gpu::open(backends) {
        Ok(gpu) => {
            let info = gpu.info();
            // Another backend counts when it opens the same card, by its vendor and device.
            let same_card = |b: Backend| {
                Gpu::open(&[b]).is_ok_and(|other| {
                    let other = other.info();
                    (other.vendor, other.device) == (info.vendor, info.device)
                })
            };
            let also: Vec<&str> = backends
                .iter()
                .filter(|&&b| b != info.backend && same_card(b))
                .map(|b| b.name())
                .collect();
            let detail = if also.is_empty() {
                info.backend.to_string()
            } else {
                format!("{} (also: {})", info.backend, also.join(", "))
            };
            let note = (info.vendor == NVIDIA).then(|| nvidia_note(cuda).to_string());
            DeviceEntry {
                kind: DeviceKind::Gpu,
                name: info.name.clone(),
                detail,
                available: true,
                chosen: false,
                note,
            }
        }
        Err(why) => DeviceEntry {
            kind: DeviceKind::Gpu,
            name: String::new(),
            detail: String::new(),
            available: false,
            chosen: false,
            note: Some(why.to_string()),
        },
    }
}

/// Without the `gpu` feature, the GPU's entry says how to get it.
#[cfg(not(feature = "gpu"))]
fn gpu_entry(_cuda: Option<bool>) -> DeviceEntry {
    DeviceEntry {
        kind: DeviceKind::Gpu,
        name: String::new(),
        detail: String::new(),
        available: false,
        chosen: false,
        note: Some("this build has no GPU support: build with the `gpu` feature".to_string()),
    }
}

/// The CPU's entry: always available, with the kernel level and thread count in force.
fn cpu_entry() -> DeviceEntry {
    let threads = threads();
    DeviceEntry {
        kind: DeviceKind::Cpu,
        name: "CPU".to_string(),
        detail: format!(
            "{}, {threads} thread{}",
            level(),
            if threads == 1 { "" } else { "s" }
        ),
        available: true,
        chosen: false,
        note: None,
    }
}

/// Marks the entry `choice` picks among `list` (best first): the first available one, or
/// the first available one of the kind asked for, or none if that kind is not available.
pub(crate) fn mark_chosen(list: &mut [DeviceEntry], choice: Choice) {
    for entry in list.iter_mut() {
        entry.chosen = false;
    }
    let pick = list.iter_mut().find(|e| {
        e.available
            && match choice {
                Choice::Best => true,
                Choice::Only(kind) => e.kind == kind,
            }
    });
    if let Some(entry) = pick {
        entry.chosen = true;
    }
}

/// The devices this machine and this build can train on, best first: with the `cuda` feature
/// an NVIDIA card through CUDA, then the GPU through wgpu, then the CPU.
/// A device that cannot be used is listed too, with the reason in its
/// [`note`](DeviceEntry::note), and the CPU is always available. The entry
/// [`Device::Auto`](crate::Device) picks is marked [`chosen`](DeviceEntry::chosen): the first
/// available one, or, when `CAVITATION_DEVICE` is `cuda`, `gpu` or `cpu`, the first available
/// one of that kind.
///
/// The choice here is the machine's. For one run, [`fit`](crate::fit) with `Device::Auto`
/// also looks at the run: it leaves what a device cannot train to the next one, the CPU
/// last, and [`History::device`](crate::History) says where the run trained.
///
/// With the `gpu` feature this opens a device on each of the platform's backends to find
/// the GPU, and with the `cuda` feature it opens CUDA, loading NVIDIA's libraries, which can
/// each take a moment; without the `gpu` feature the GPU's entry says the build has no GPU
/// support, and without the `cuda` feature there is no CUDA entry. A GPU is listed before the
/// CPU whatever the model: on a small network, where a step's upload, launches and readback
/// weigh as much as its arithmetic, it may gain nothing over the CPU.
///
/// # Errors
///
/// [`Error::InvalidData`] if `CAVITATION_DEVICE` is set to something other than `auto`,
/// `cuda`, `gpu` or `cpu`.
///
/// # Panics
///
/// As `cavitation_core::kernels::level` and `threads` do, for the CPU's entry: if
/// `CAVITATION_KERNEL` or `CAVITATION_THREADS` holds a value they do not take.
///
/// # Examples
///
/// ```
/// use cavitation_nn::{DeviceKind, devices};
///
/// // An error only if `CAVITATION_DEVICE` holds something other than `auto`, `cuda`, `gpu`
/// // or `cpu`.
/// if let Ok(list) = devices() {
///     for entry in &list {
///         println!("{entry}");
///     }
///     // The CPU is last and always available, and at most one entry is chosen.
///     assert_eq!(list.last().map(|e| (e.kind, e.available)), Some((DeviceKind::Cpu, true)));
///     assert!(list.iter().filter(|e| e.chosen).count() <= 1);
/// }
/// ```
pub fn devices() -> Result<Vec<DeviceEntry>, Error> {
    let choice = choice_from_env()?;
    let mut list: Vec<DeviceEntry> = Vec::with_capacity(3);
    #[cfg(feature = "cuda")]
    list.push(cuda_entry());
    let cuda = list.first().map(|e| e.available);
    list.push(gpu_entry(cuda));
    list.push(cpu_entry());
    mark_chosen(&mut list, choice);
    Ok(list)
}

#[cfg(test)]
mod tests {
    //! Choosing a device: `CAVITATION_DEVICE` and the order `Device::Auto` tries the devices in.
    use super::*;

    /// An entry of `kind`, available or not.
    fn entry(kind: DeviceKind, available: bool) -> DeviceEntry {
        DeviceEntry {
            kind,
            name: "name".to_string(),
            detail: "detail".to_string(),
            available,
            chosen: false,
            note: (!available).then(|| "why".to_string()),
        }
    }

    /// The kinds marked chosen in `list`.
    fn chosen(list: &[DeviceEntry]) -> Vec<DeviceKind> {
        list.iter().filter(|e| e.chosen).map(|e| e.kind).collect()
    }

    /// Unset, empty, blank and `auto` in any case ask for the best device; `cpu`, `gpu` and
    /// `cuda` for that kind; anything else is an error naming the variable, the value and
    /// the values accepted.
    #[test]
    fn the_variable_takes_auto_cpu_gpu_and_cuda() {
        for best in [None, Some(""), Some("  "), Some("auto"), Some(" AUTO ")] {
            assert_eq!(parse_choice(best).unwrap(), Choice::Best, "{best:?}");
        }
        assert_eq!(
            parse_choice(Some("cpu")).unwrap(),
            Choice::Only(DeviceKind::Cpu)
        );
        assert_eq!(
            parse_choice(Some(" Gpu ")).unwrap(),
            Choice::Only(DeviceKind::Gpu)
        );
        assert_eq!(
            parse_choice(Some("CUDA")).unwrap(),
            Choice::Only(DeviceKind::Cuda)
        );
        let message = parse_choice(Some("tpu")).unwrap_err().to_string();
        assert!(
            message.contains("CAVITATION_DEVICE: unknown device \"tpu\"")
                && message.contains("expected auto, cuda, gpu or cpu"),
            "{message}"
        );
    }

    /// The best device is the first available one; a kind asked for is chosen when it is
    /// available and nothing is when it is not; a second marking replaces the first.
    #[test]
    fn the_first_available_entry_of_the_kind_is_chosen() {
        use DeviceKind::{Cpu, Gpu};
        let mut both = vec![entry(Gpu, true), entry(Cpu, true)];
        mark_chosen(&mut both, Choice::Best);
        assert_eq!(chosen(&both), [Gpu]);
        mark_chosen(&mut both, Choice::Only(Cpu));
        assert_eq!(chosen(&both), [Cpu]);
        mark_chosen(&mut both, Choice::Only(Gpu));
        assert_eq!(chosen(&both), [Gpu]);

        let mut cpu_only = vec![entry(Gpu, false), entry(Cpu, true)];
        mark_chosen(&mut cpu_only, Choice::Best);
        assert_eq!(chosen(&cpu_only), [Cpu]);
        mark_chosen(&mut cpu_only, Choice::Only(Gpu));
        assert_eq!(chosen(&cpu_only), []);
    }

    /// The list ends with an available CPU whose detail names the kernel level and the
    /// threads, has the GPU before it and CUDA (with the `cuda` feature) first, and (with the
    /// variable unset or valid) chooses at most one entry, an available one.
    #[test]
    fn the_list_ends_with_the_cpu() {
        let Ok(list) = devices() else {
            // `CAVITATION_DEVICE` holds something else in this environment.
            return;
        };
        let kinds: Vec<DeviceKind> = list.iter().map(|e| e.kind).collect();
        if cfg!(feature = "cuda") {
            assert_eq!(kinds, [DeviceKind::Cuda, DeviceKind::Gpu, DeviceKind::Cpu]);
        } else {
            assert_eq!(kinds, [DeviceKind::Gpu, DeviceKind::Cpu]);
        }
        let cpu = list.last().unwrap();
        assert!(cpu.available && cpu.note.is_none());
        assert!(
            cpu.detail.starts_with(&level().to_string()) && cpu.detail.contains("thread"),
            "{}",
            cpu.detail
        );
        let picked: Vec<&DeviceEntry> = list.iter().filter(|e| e.chosen).collect();
        assert!(picked.len() <= 1 && picked.iter().all(|e| e.available));
        // A device that is not available says why.
        assert!(list.iter().all(|e| e.available || e.note.is_some()));
    }

    /// An entry's line: the kind, the name, the detail and the note, or the reason it is not
    /// available, and the mark on the chosen one.
    #[test]
    fn an_entry_prints_on_one_line() {
        let mut gpu = entry(DeviceKind::Gpu, true);
        gpu.note = Some("a note".to_string());
        gpu.chosen = true;
        assert_eq!(gpu.to_string(), "gpu  name, detail (a note)  <- chosen");
        assert_eq!(
            entry(DeviceKind::Cpu, true).to_string(),
            "cpu  name, detail"
        );
        assert_eq!(
            entry(DeviceKind::Gpu, false).to_string(),
            "gpu  not available: why"
        );
    }
}
