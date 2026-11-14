//! Runs the numpy baseline script and parses the JSON line it prints, and chooses the BLAS
//! settings numpy is timed at: its defaults, or with `--numpy-best` also one BLAS thread and,
//! where OpenBLAS picked kernels older than the CPU can run, the best kernels it can.

use std::path::PathBuf;
use std::process::Command;

/// What `scripts/bench/numpy_mlp.py` measured.
#[derive(Clone, Debug, PartialEq)]
pub struct NumpyResult {
    /// `train` or `infer`.
    pub kind: String,
    /// Steps per epoch (train) or calls per round (infer).
    pub units: f64,
    /// Seconds per timed epoch or round.
    pub samples: Vec<f64>,
    /// The median of `samples`, in seconds.
    pub median: f64,
    /// The health warnings the script's checks raised (0 for inference); `None` from a
    /// script that does not report them.
    pub warnings: Option<usize>,
    /// The numpy version that ran the script.
    pub numpy_version: String,
    /// The BLAS numpy ran on: library, version, OpenBLAS core and thread count.
    pub blas: String,
    /// The OpenBLAS core (kernel set) name, or `unknown`.
    pub blas_core: String,
}

impl NumpyResult {
    /// Checks that the script did the work asked for: the same `kind`, `units` (steps per
    /// epoch or calls per round) and number of timed samples (epochs or rounds).
    ///
    /// # Errors
    ///
    /// Returns a message naming both values for the first mismatch.
    pub fn check(&self, kind: &str, units: usize, samples: usize) -> Result<(), String> {
        if self.kind != kind {
            return Err(format!("numpy ran `{}`, expected `{kind}`", self.kind));
        }
        if self.units != units as f64 {
            return Err(format!(
                "numpy did {} steps or calls per sample, cavitation did {units}",
                self.units
            ));
        }
        if self.samples.len() != samples {
            return Err(format!(
                "numpy timed {} epochs or rounds, cavitation timed {samples}",
                self.samples.len()
            ));
        }
        Ok(())
    }
}

/// The text after `"key":` in a flat JSON object on one line, up to the next top-level comma
/// or the closing brace. Arrays and strings (with their quotes, commas inside and escaped
/// quotes) stay whole.
fn raw_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = line[start..].trim_start();
    let end = if rest.starts_with('[') {
        rest.find(']')? + 1
    } else if let Some(body) = rest.strip_prefix('"') {
        let mut escaped = false;
        let close = body.char_indices().find_map(|(i, c)| {
            let found = c == '"' && !escaped;
            escaped = c == '\\' && !escaped;
            found.then_some(i)
        })?;
        close + 2
    } else {
        rest.find([',', '}'])?
    };
    Some(rest[..end].trim())
}

/// The text of a JSON string value (quotes included), with `\"` and `\\` unescaped; other
/// escapes are kept as written.
fn json_text(raw: &str) -> String {
    let body = raw
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(raw);
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match (c, chars.clone().next()) {
            ('\\', Some(next @ ('"' | '\\'))) => {
                out.push(next);
                chars.next();
            }
            _ => out.push(c),
        }
    }
    out
}

/// Parses one JSON line of the script's output.
fn parse_line(line: &str) -> Option<NumpyResult> {
    let kind = raw_value(line, "kind")?.trim_matches('"').to_string();
    let units = raw_value(line, "units")?.parse().ok()?;
    let median = raw_value(line, "median")?.parse().ok()?;
    let list = raw_value(line, "samples")?
        .trim_start_matches('[')
        .trim_end_matches(']');
    let samples = list
        .split(',')
        .map(|v| v.trim().parse().ok())
        .collect::<Option<Vec<f64>>>()?;
    let text = |key| raw_value(line, key).map(json_text);
    let unknown = || "unknown".to_string();
    Some(NumpyResult {
        warnings: raw_value(line, "warnings").and_then(|v| v.parse().ok()),
        numpy_version: text("numpy_version").unwrap_or_else(unknown),
        blas: text("blas").unwrap_or_else(unknown),
        blas_core: text("blas_core").unwrap_or_else(unknown),
        kind,
        units,
        samples,
        median,
    })
}

/// Finds the JSON line (the last line starting with `{`) in the script's standard output
/// and parses it.
///
/// # Errors
///
/// Returns a message if no line parses as the script's result.
pub fn parse_output(output: &str) -> Result<NumpyResult, String> {
    output
        .lines()
        .rev()
        .filter(|l| l.trim_start().starts_with('{'))
        .find_map(|l| parse_line(l.trim()))
        .ok_or_else(|| format!("no result line in the numpy script's output:\n{output}"))
}

/// The path of the baseline script in the repository checkout.
fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/bench/numpy_mlp.py")
}

/// Advice appended to every failure to run the baseline: the script runs under `python -I`,
/// which does not see user-site packages.
const NUMPY_ADVICE: &str = "The baseline runs under `python3 -I`, which does not see \
    user-site packages (`pip install --user`). Install numpy into a virtual environment and \
    pass its interpreter with `--python <path>` or `CAVITATION_BENCH_PYTHON`.";

/// Runs `python -I scripts/bench/numpy_mlp.py <args>` with the environment variables `env`
/// added, and parses its result.
///
/// # Errors
///
/// Returns a message if the interpreter cannot start, exits with a failure (for example
/// `ModuleNotFoundError: No module named 'numpy'`, shown from its standard error), or prints
/// no result line. The first two messages end with advice: the script runs under `-I`,
/// which does not see user-site packages, so numpy belongs in a virtual environment whose
/// interpreter is passed with `--python <path>` or `CAVITATION_BENCH_PYTHON`.
pub fn run(python: &str, args: &[String], env: &[(String, String)]) -> Result<NumpyResult, String> {
    let out = Command::new(python)
        .arg("-I")
        .arg(script_path())
        .args(args)
        .envs(env.iter().map(|(k, v)| (k, v)))
        .output()
        .map_err(|e| format!("could not run `{python}`: {e}\n{NUMPY_ADVICE}"))?;
    if !out.status.success() {
        return Err(format!(
            "`{python} -I numpy_mlp.py` failed ({}):\n{}\n{NUMPY_ADVICE}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim_end()
        ));
    }
    parse_output(&String::from_utf8_lossy(&out.stdout))
}

/// The variables that set a BLAS to one thread: OpenBLAS's, OpenMP's (OpenBLAS built with
/// OpenMP, and BLIS), MKL's, and Apple Accelerate's.
const ONE_THREAD: [&str; 4] = [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
];

/// The x86 instruction sets that decide which OpenBLAS kernels a CPU can run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    /// SSE4.2.
    pub sse42: bool,
    /// AVX.
    pub avx: bool,
    /// AVX2.
    pub avx2: bool,
    /// FMA (the three-operand fused multiply-add of Haswell and later).
    pub fma: bool,
}

/// This CPU's features, from the standard library's run-time detection; `None` on a CPU
/// that is not x86-64, where OpenBLAS's choice is left alone.
pub fn cpu_features() -> Option<CpuFeatures> {
    #[cfg(target_arch = "x86_64")]
    {
        Some(CpuFeatures {
            sse42: std::arch::is_x86_feature_detected!("sse4.2"),
            avx: std::arch::is_x86_feature_detected!("avx"),
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            fma: std::arch::is_x86_feature_detected!("fma"),
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// The newest OpenBLAS core (`OPENBLAS_CORETYPE`) a CPU with `features` can run, chosen
/// conservatively: `Haswell` needs AVX2 and FMA, `Sandybridge` AVX, `Nehalem` SSE4.2.
/// AVX-512 cores (`SkylakeX` and later) are never chosen, since some CPUs slow their clock
/// for them. `None` below SSE4.2.
pub fn openblas_target(features: CpuFeatures) -> Option<&'static str> {
    if features.avx && features.avx2 && features.fma {
        Some("Haswell")
    } else if features.avx {
        Some("Sandybridge")
    } else if features.sse42 {
        Some("Nehalem")
    } else {
        None
    }
}

/// Where an x86 OpenBLAS core name stands: 0 for cores older than SSE4.2's (`Katmai` up to
/// `Core2`, `Penryn` and `Atom`, and the generic C kernels), 1 for `Nehalem`, 2 for the AVX
/// cores, 3 for AVX2 and later. `None` for a name not listed (case is ignored).
pub fn core_rank(name: &str) -> Option<u8> {
    let rank = match name.to_ascii_lowercase().as_str() {
        "generic" | "katmai" | "coppermine" | "northwood" | "prescott" | "banias" | "atom"
        | "athlon" | "opteron" | "opteron_sse3" | "barcelona" | "bobcat" | "nano" | "core2"
        | "penryn" | "dunnington" => 0,
        "nehalem" => 1,
        "sandybridge" | "bulldozer" | "piledriver" | "steamroller" | "excavator" => 2,
        "haswell" | "zen" | "skylakex" | "cooperlake" | "sapphirerapids" => 3,
        _ => return None,
    };
    Some(rank)
}

/// The OpenBLAS core to set instead of `reported`, if `reported` is a known core older than
/// `target`; `None` when the core is unknown, already good enough, or there is no target.
pub fn core_upgrade(reported: &str, target: Option<&'static str>) -> Option<&'static str> {
    let target = target?;
    (core_rank(reported)? < core_rank(target)?).then_some(target)
}

/// One way of running numpy: a name for the reports and the variables that set it up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Setting {
    /// For example `default`, `1 thread` or `core Nehalem, 1 thread`.
    pub label: String,
    /// The variables added to the child's environment.
    pub env: Vec<(String, String)>,
}

/// The settings `--numpy-best` tries after the default one: one thread; and, when `upgrade`
/// names a core, that core at the default thread count and at one thread.
pub fn best_settings(upgrade: Option<&str>) -> Vec<Setting> {
    let one: Vec<(String, String)> = ONE_THREAD
        .iter()
        .map(|k| (k.to_string(), "1".to_string()))
        .collect();
    let mut settings = vec![Setting {
        label: "1 thread".to_string(),
        env: one.clone(),
    }];
    if let Some(core) = upgrade {
        let core_env = ("OPENBLAS_CORETYPE".to_string(), core.to_string());
        settings.push(Setting {
            label: format!("core {core}"),
            env: vec![core_env.clone()],
        });
        settings.push(Setting {
            label: format!("core {core}, 1 thread"),
            env: [vec![core_env], one].concat(),
        });
    }
    settings
}

/// numpy's runs of one comparison: at the default settings, and at the fastest one tried.
#[derive(Clone, Debug)]
pub struct NumpyRuns {
    /// The run at numpy's default settings.
    pub default: NumpyResult,
    /// The fastest run (the default one unless `--numpy-best` found a faster setting).
    pub best: NumpyResult,
    /// The setting of `best`.
    pub best_label: String,
    /// Every setting tried, in order, with its median seconds.
    pub tried: Vec<(String, f64)>,
}

/// The label of a run at `setting`: its own, marked when OpenBLAS did not take the core it
/// was asked for (a core left out of its build), since that run is the default core again.
fn checked_label(setting: &Setting, result: &NumpyResult) -> String {
    match setting.env.iter().find(|(k, _)| k == "OPENBLAS_CORETYPE") {
        Some((_, core)) if !result.blas_core.eq_ignore_ascii_case(core) => format!(
            "{} (not taken: OpenBLAS runs {})",
            setting.label, result.blas_core
        ),
        _ => setting.label.clone(),
    }
}

/// Runs the baseline at numpy's default settings and, if `best`, at each of
/// [`best_settings`] (the core upgrade from the default run's core and this CPU's
/// features), each run checked with `check`; returns them with the fastest picked by median.
///
/// # Errors
///
/// Returns a message if a run fails or does not pass `check`.
pub fn run_settings(
    python: &str,
    args: &[String],
    best: bool,
    check: impl Fn(&NumpyResult) -> Result<(), String>,
) -> Result<NumpyRuns, String> {
    let default = run(python, args, &[])?;
    check(&default)?;
    let mut runs = NumpyRuns {
        best: default.clone(),
        best_label: "default".to_string(),
        tried: vec![("default".to_string(), default.median)],
        default,
    };
    if !best {
        return Ok(runs);
    }
    let upgrade = core_upgrade(
        &runs.default.blas_core,
        cpu_features().and_then(openblas_target),
    );
    for setting in best_settings(upgrade) {
        eprintln!("  numpy at {}", setting.label);
        let result = run(python, args, &setting.env)?;
        check(&result)?;
        let label = checked_label(&setting, &result);
        runs.tried.push((label.clone(), result.median));
        if result.median < runs.best.median {
            runs.best = result;
            runs.best_label = label;
        }
    }
    Ok(runs)
}

/// Tests for the baseline's output parser.
#[cfg(test)]
mod tests {
    use super::*;

    /// One JSON line among other output gives the kind, the sample times and the median.
    #[test]
    fn numpy_json_line_parsed() {
        let out = "numpy 2.4.4\n{\"kind\": \"train\", \"units\": 469, \
            \"samples\": [3.5, 3.25, 3.75], \"median\": 3.5}\nbye\n";
        let r = parse_output(out).unwrap();
        assert_eq!(r.kind, "train");
        assert_eq!(r.samples, [3.5, 3.25, 3.75]);
        assert_eq!(r.median, 3.5);
        assert_eq!(r.units, 469.0);
        let quoted = "{\"kind\": \"infer\", \"blas\": \"a, \\\"b\\\" c:\\\\d\", \"units\": 1, \
            \"samples\": [2.0], \"median\": 2.0}";
        assert_eq!(parse_output(quoted).unwrap().blas, "a, \"b\" c:\\d");
        assert_eq!(r.numpy_version, "unknown");
        assert_eq!(
            (r.blas.as_str(), r.blas_core.as_str()),
            ("unknown", "unknown")
        );
        assert_eq!(r.warnings, None);
        assert!(parse_output("no json here").is_err());
        assert!(parse_output("{\"kind\": \"train\"}").is_err());
    }

    /// A result with the wrong kind, steps or sample count is rejected, naming both values.
    #[test]
    fn mismatched_numpy_work_rejected() {
        let out = "{\"kind\": \"train\", \"units\": 469, \"samples\": [1.0, 2.0, 3.0], \
            \"median\": 2.0, \"warnings\": 6, \"numpy_version\": \"2.4.4\", \
            \"blas\": \"openblas 0.3.34, core Katmai, 4 threads (pthreads)\", \
            \"blas_core\": \"Katmai\"}";
        let r = parse_output(out).unwrap();
        assert_eq!(r.numpy_version, "2.4.4");
        assert_eq!(r.blas, "openblas 0.3.34, core Katmai, 4 threads (pthreads)");
        assert_eq!((r.blas_core.as_str(), r.warnings), ("Katmai", Some(6)));
        assert!(r.check("train", 469, 3).is_ok());
        assert!(
            r.check("infer", 469, 3)
                .unwrap_err()
                .contains("`train`, expected `infer`")
        );
        assert!(
            r.check("train", 1875, 3)
                .unwrap_err()
                .contains("469 steps or calls per sample, cavitation did 1875")
        );
        assert!(
            r.check("train", 469, 4)
                .unwrap_err()
                .contains("timed 3 epochs or rounds, cavitation timed 4")
        );
    }

    /// An interpreter that exits with a failure, or does not exist, gets advice on making
    /// numpy visible to the baseline.
    #[test]
    fn failed_baseline_explains_how_to_make_numpy_visible() {
        for python in ["false", "no-such-python-interpreter"] {
            let err = run(python, &[], &[]).unwrap_err();
            assert!(err.contains("python3 -I"), "{err}");
            assert!(err.contains("pip install --user"), "{err}");
            assert!(err.contains("virtual environment"), "{err}");
            assert!(err.contains("--python <path>"), "{err}");
            assert!(err.contains("CAVITATION_BENCH_PYTHON"), "{err}");
        }
    }

    /// The CPU's features map to the newest core it can run, never an AVX-512 one.
    #[test]
    fn cpu_features_map_to_openblas_target() {
        let f = |sse42, avx, avx2, fma| CpuFeatures {
            sse42,
            avx,
            avx2,
            fma,
        };
        assert_eq!(openblas_target(f(false, false, false, false)), None);
        assert_eq!(
            openblas_target(f(true, false, false, false)),
            Some("Nehalem")
        );
        assert_eq!(
            openblas_target(f(true, true, false, false)),
            Some("Sandybridge")
        );
        // AVX2 without FMA does not run the Haswell kernels.
        assert_eq!(
            openblas_target(f(true, true, true, false)),
            Some("Sandybridge")
        );
        assert_eq!(openblas_target(f(true, true, true, true)), Some("Haswell"));
    }

    /// The core is upgraded only from a known older core: an SSE4.2 CPU on Katmai gets
    /// Nehalem, an AVX2 CPU already on Haswell or Zen keeps it, an unknown core is left.
    #[test]
    fn core_upgraded_only_when_known_older() {
        assert_eq!(core_upgrade("Katmai", Some("Nehalem")), Some("Nehalem"));
        assert_eq!(core_upgrade("prescott", Some("Haswell")), Some("Haswell"));
        assert_eq!(
            core_upgrade("Sandybridge", Some("Haswell")),
            Some("Haswell")
        );
        assert_eq!(core_upgrade("Nehalem", Some("Nehalem")), None);
        assert_eq!(core_upgrade("Haswell", Some("Haswell")), None);
        assert_eq!(core_upgrade("Zen", Some("Haswell")), None);
        assert_eq!(core_upgrade("SkylakeX", Some("Haswell")), None);
        assert_eq!(core_upgrade("unknown", Some("Haswell")), None);
        assert_eq!(core_upgrade("NeoverseN1", Some("Haswell")), None);
        assert_eq!(core_upgrade("Katmai", None), None);
    }

    /// `--numpy-best` tries one thread, and with a core upgrade that core at both thread
    /// counts; one thread is set for every BLAS numpy may be built on.
    #[test]
    fn best_settings_cover_threads_and_core() {
        let plain = best_settings(None);
        assert_eq!(plain.len(), 1);
        assert_eq!(plain[0].label, "1 thread");
        for key in ONE_THREAD {
            assert!(plain[0].env.contains(&(key.to_string(), "1".to_string())));
        }
        let all = best_settings(Some("Nehalem"));
        let labels: Vec<&str> = all.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(
            labels,
            ["1 thread", "core Nehalem", "core Nehalem, 1 thread"]
        );
        let core = ("OPENBLAS_CORETYPE".to_string(), "Nehalem".to_string());
        assert_eq!(all[1].env, std::slice::from_ref(&core));
        assert!(all[2].env.contains(&core) && all[2].env.len() == 1 + ONE_THREAD.len());
    }

    /// A run whose OpenBLAS did not take the core asked for says so in its label.
    #[test]
    fn core_not_taken_is_labelled() {
        let out = "{\"kind\": \"infer\", \"units\": 1, \"samples\": [1.0], \"median\": 1.0, \
            \"blas_core\": \"Katmai\"}";
        let result = parse_output(out).unwrap();
        let setting = &best_settings(Some("Nehalem"))[1];
        assert_eq!(
            checked_label(setting, &result),
            "core Nehalem (not taken: OpenBLAS runs Katmai)"
        );
        let taken = NumpyResult {
            blas_core: "nehalem".to_string(),
            ..result
        };
        assert_eq!(checked_label(setting, &taken), "core Nehalem");
    }
}
