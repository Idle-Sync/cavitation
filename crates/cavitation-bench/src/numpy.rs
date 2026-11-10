//! Runs the numpy baseline script and parses the JSON line it prints.

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
    /// The numpy version that ran the script.
    pub numpy_version: String,
    /// The BLAS thread setting the script saw.
    pub blas_threads: String,
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
/// or the closing brace. Arrays stay whole.
fn raw_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\":");
    let start = line.find(&marker)? + marker.len();
    let rest = line[start..].trim_start();
    let end = if rest.starts_with('[') {
        rest.find(']')? + 1
    } else {
        rest.find([',', '}'])?
    };
    Some(rest[..end].trim())
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
    let text = |key| raw_value(line, key).map(|v| v.trim_matches('"').to_string());
    Some(NumpyResult {
        numpy_version: text("numpy_version").unwrap_or_else(|| "unknown".to_string()),
        blas_threads: text("blas_threads").unwrap_or_else(|| "unknown".to_string()),
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

/// Runs `python -I scripts/bench/numpy_mlp.py <args>` and parses its result.
///
/// # Errors
///
/// Returns a message if the interpreter cannot start (for example numpy missing is reported
/// from its standard error), exits with a failure, or prints no result line.
pub fn run(python: &str, args: &[String]) -> Result<NumpyResult, String> {
    let out = Command::new(python)
        .arg("-I")
        .arg(script_path())
        .args(args)
        .output()
        .map_err(|e| {
            format!(
                "could not run `{python}`: {e}; pass --python <path> or set CAVITATION_BENCH_PYTHON"
            )
        })?;
    if !out.status.success() {
        return Err(format!(
            "`{python} -I numpy_mlp.py` failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse_output(&String::from_utf8_lossy(&out.stdout))
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
        assert_eq!(r.numpy_version, "unknown");
        assert!(parse_output("no json here").is_err());
        assert!(parse_output("{\"kind\": \"train\"}").is_err());
    }

    /// A result with the wrong kind, steps or sample count is rejected, naming both values.
    #[test]
    fn mismatched_numpy_work_rejected() {
        let out = "{\"kind\": \"train\", \"units\": 469, \"samples\": [1.0, 2.0, 3.0], \
            \"median\": 2.0, \"numpy_version\": \"2.4.4\", \"blas_threads\": \"x:4\"}";
        let r = parse_output(out).unwrap();
        assert_eq!(
            (r.numpy_version.as_str(), r.blas_threads.as_str()),
            ("2.4.4", "x:4")
        );
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
}
