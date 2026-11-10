//! The machine a benchmark ran on: CPU model, logical cores, compiler version.

use std::process::Command;

/// The CPU a benchmark ran on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineInfo {
    /// The CPU's marketing name, for example `Intel(R) Celeron(R) N5105 @ 2.00GHz`.
    pub cpu_model: String,
    /// The number of logical cores (hardware threads).
    pub logical_cores: usize,
}

/// Reads the CPU model and logical core count from the text of `/proc/cpuinfo`: the first
/// `model name` line, and the number of `processor` lines. Returns `None` if either is
/// missing.
pub fn parse_cpuinfo(text: &str) -> Option<MachineInfo> {
    let mut model = None;
    let mut cores = 0;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "model name" if model.is_none() => model = Some(value.trim().to_string()),
            "processor" => cores += 1,
            _ => {}
        }
    }
    let cpu_model = model.filter(|m| !m.is_empty())?;
    (cores > 0).then_some(MachineInfo {
        cpu_model,
        logical_cores: cores,
    })
}

/// Builds the description from the output of `sysctl -n machdep.cpu.brand_string` and
/// `sysctl -n hw.ncpu` (macOS). Returns `None` if the name is empty or the count is not a
/// positive integer.
pub fn parse_sysctl(brand: &str, ncpu: &str) -> Option<MachineInfo> {
    let cpu_model = brand.trim().to_string();
    let logical_cores: usize = ncpu.trim().parse().ok().filter(|&n| n > 0)?;
    (!cpu_model.is_empty()).then_some(MachineInfo {
        cpu_model,
        logical_cores,
    })
}

/// Runs `sysctl -n key` and returns its output, if the command works.
fn sysctl(key: &str) -> Option<String> {
    let out = Command::new("sysctl").args(["-n", key]).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Describes this machine: `/proc/cpuinfo` on Linux, `sysctl` on macOS. Falls back to an
/// unknown model and the core count the standard library reports.
pub fn detect() -> MachineInfo {
    let found = if cfg!(target_os = "macos") {
        parse_sysctl(
            &sysctl("machdep.cpu.brand_string").unwrap_or_default(),
            &sysctl("hw.ncpu").unwrap_or_default(),
        )
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| parse_cpuinfo(&text))
    };
    found.unwrap_or_else(|| MachineInfo {
        cpu_model: "unknown".to_string(),
        logical_cores: std::thread::available_parallelism().map_or(1, usize::from),
    })
}

/// The version line of the compiler on `PATH` (`rustc --version`), or `unknown`.
pub fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Tests for the machine description parsers.
#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed sample of `/proc/cpuinfo`: four logical processors of one CPU model.
    const CPUINFO: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\n\
        model name\t: Intel(R) Celeron(R) N5105 @ 2.00GHz\ncpu MHz\t\t: 2000.000\n\n\
        processor\t: 1\nvendor_id\t: GenuineIntel\n\
        model name\t: Intel(R) Celeron(R) N5105 @ 2.00GHz\n\n\
        processor\t: 2\nmodel name\t: Intel(R) Celeron(R) N5105 @ 2.00GHz\n\n\
        processor\t: 3\nmodel name\t: Intel(R) Celeron(R) N5105 @ 2.00GHz\n";

    /// The CPU model is the first `model name` line and the cores are the `processor` lines.
    #[test]
    fn machine_info_parses_proc_cpuinfo() {
        let info = parse_cpuinfo(CPUINFO).unwrap();
        assert_eq!(info.cpu_model, "Intel(R) Celeron(R) N5105 @ 2.00GHz");
        assert_eq!(info.logical_cores, 4);
        assert!(parse_cpuinfo("nothing useful\n").is_none());
    }

    /// The macOS values come from `sysctl -n machdep.cpu.brand_string` and `hw.ncpu`.
    #[test]
    fn machine_info_parses_sysctl() {
        let info = parse_sysctl("Apple M4\n", "10\n").unwrap();
        assert_eq!(info.cpu_model, "Apple M4");
        assert_eq!(info.logical_cores, 10);
        assert!(parse_sysctl("Apple M4", "ten").is_none());
        assert!(parse_sysctl("", "10").is_none());
    }
}
