//! The machine a benchmark ran on: CPU model, logical cores, compiler version, and the CPU
//! time this process has used.

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

/// The processor name in the output of
/// `reg query HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0 /v ProcessorNameString`
/// (Windows): the text after `REG_SZ` on the `ProcessorNameString` line, trimmed. Returns
/// `None` if there is no such line or the name is empty.
pub fn parse_reg_processor_name(text: &str) -> Option<String> {
    text.lines()
        .filter(|line| line.trim_start().starts_with("ProcessorNameString"))
        .find_map(|line| line.split_once("REG_SZ"))
        .map(|(_, name)| name.trim().to_string())
        .filter(|name| !name.is_empty())
}

/// Runs `program` with `args` and returns its standard output, if the command works.
fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Describes this machine: `/proc/cpuinfo` on Linux, `sysctl` on macOS, the registry's
/// processor name (through `reg query`) on Windows. Falls back to an unknown model and the
/// core count the standard library reports.
pub fn detect() -> MachineInfo {
    let found = if cfg!(target_os = "macos") {
        parse_sysctl(
            &sysctl("machdep.cpu.brand_string").unwrap_or_default(),
            &sysctl("hw.ncpu").unwrap_or_default(),
        )
    } else if cfg!(windows) {
        run(
            "reg",
            &[
                "query",
                r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "/v",
                "ProcessorNameString",
            ],
        )
        .and_then(|text| parse_reg_processor_name(&text))
        .map(|cpu_model| MachineInfo {
            cpu_model,
            logical_cores: std::thread::available_parallelism().map_or(1, usize::from),
        })
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

/// The three load averages (1, 5 and 15 minutes) from the text of `/proc/loadavg`
/// (`0.52 0.58 0.59 1/467 1234`) or of `sysctl -n vm.loadavg` (`{ 1.79 2.10 2.30 }`), joined
/// by single spaces. Returns `None` if the text does not start with three numbers.
pub fn parse_load_average(text: &str) -> Option<String> {
    let values: Vec<&str> = text
        .split_whitespace()
        .filter(|word| *word != "{" && *word != "}")
        .take(3)
        .collect();
    (values.len() == 3 && values.iter().all(|v| v.parse::<f64>().is_ok())).then(|| values.join(" "))
}

/// The CPU busy percentage in the output of
/// `powershell -NoProfile -Command (Get-CimInstance Win32_Processor).LoadPercentage`: the
/// whole numbers it prints (one per processor package), averaged and written as `busy N%`.
/// Returns `None` if it prints no number.
pub fn parse_load_percentage(text: &str) -> Option<String> {
    let values: Vec<f64> = text
        .split_whitespace()
        .filter_map(|word| word.parse::<f64>().ok())
        .collect();
    (!values.is_empty()).then(|| {
        format!(
            "busy {:.0}%",
            values.iter().sum::<f64>() / values.len() as f64
        )
    })
}

/// The load now: the load averages from `/proc/loadavg` on Linux and `sysctl -n vm.loadavg`
/// on macOS; on Windows, which has no load average, the CPU busy percentage over the last
/// moment (`Win32_Processor.LoadPercentage`); `unknown` where none can be read.
pub fn load_average() -> String {
    let found = if cfg!(target_os = "macos") {
        sysctl("vm.loadavg").and_then(|t| parse_load_average(&t))
    } else if cfg!(windows) {
        run(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).LoadPercentage",
            ],
        )
        .and_then(|t| parse_load_percentage(&t))
    } else {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|t| parse_load_average(&t))
    };
    found.unwrap_or_else(|| "unknown".to_string())
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

/// Clock ticks per second in `/proc/<pid>/stat`'s time fields (`USER_HZ`), which Linux
/// fixes at 100 for user space on x86 and Arm.
const TICKS_PER_SECOND: f64 = 100.0;

/// The user plus system CPU seconds in the text of `/proc/<pid>/stat`: fields 14 (`utime`)
/// and 15 (`stime`), counted after the parenthesised command name (which may hold spaces).
/// Returns `None` if the text does not have them.
pub fn parse_proc_stat(text: &str) -> Option<f64> {
    let (_, after_name) = text.rsplit_once(')')?;
    // After the name come field 3 (the state) onwards, so `utime` is the 12th value.
    let mut fields = after_name.split_whitespace().skip(11);
    let user: u64 = fields.next()?.parse().ok()?;
    let system: u64 = fields.next()?.parse().ok()?;
    Some((user + system) as f64 / TICKS_PER_SECOND)
}

/// The CPU seconds (user plus system, all threads) this process has used so far, read from
/// `/proc/self/stat` to the nearest 10 ms; `None` where that file does not exist (anything
/// but Linux) or cannot be parsed.
pub fn process_cpu_seconds() -> Option<f64> {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|text| parse_proc_stat(&text))
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

    /// The processor name comes from the `ProcessorNameString` line of `reg query`'s output,
    /// spaces in the name kept; output without that line, or with an empty name, gives `None`.
    #[test]
    fn registry_processor_name_parses() {
        let out = "\r\nHKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\r\n    \
                   ProcessorNameString    REG_SZ    13th Gen Intel(R) Core(TM) i5-1335U\r\n\r\n";
        assert_eq!(
            parse_reg_processor_name(out),
            Some("13th Gen Intel(R) Core(TM) i5-1335U".to_string())
        );
        assert_eq!(parse_reg_processor_name("ERROR: not found"), None);
        assert_eq!(
            parse_reg_processor_name("    ProcessorNameString    REG_SZ    \r\n"),
            None
        );
    }

    /// The Windows load is the mean of the busy percentages printed, one per package; no
    /// number gives `None`.
    #[test]
    fn load_percentage_parses() {
        assert_eq!(
            parse_load_percentage("37\r\n"),
            Some("busy 37%".to_string())
        );
        assert_eq!(
            parse_load_percentage("20\r\n42\r\n"),
            Some("busy 31%".to_string())
        );
        assert_eq!(parse_load_percentage(""), None);
    }

    /// Both load-average formats give the three numbers.
    #[test]
    fn load_average_parses_both_formats() {
        let expected = Some("1.79 2.10 2.30".to_string());
        assert_eq!(parse_load_average("{ 1.79 2.10 2.30 }\n"), expected);
        assert_eq!(parse_load_average("1.79 2.10 2.30 1/467 1234"), expected);
        assert_eq!(parse_load_average("{ 1.79 2.10 }"), None);
        assert_eq!(parse_load_average("{ a b c }"), None);
    }

    /// The CPU time is `utime + stime` in ticks of 10 ms, found after the command name even
    /// when the name holds spaces and parentheses.
    #[test]
    fn process_cpu_seconds_parses_proc_stat() {
        let stat = "4242 (my (odd) bench) R 1 4242 4242 0 -1 4194304 120 0 0 0 \
                    250 37 0 0 20 0 5 0 100 1000 200 18446744073709551615";
        assert_eq!(parse_proc_stat(stat), Some(2.87));
        assert_eq!(parse_proc_stat("4242 (short) R 1 2"), None);
        assert_eq!(parse_proc_stat("no name here"), None);
        if cfg!(target_os = "linux") {
            assert!(process_cpu_seconds().is_some_and(|s| s >= 0.0));
        }
    }
}
