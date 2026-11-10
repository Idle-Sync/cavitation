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
    Some(NumpyResult {
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
        assert!(parse_output("no json here").is_err());
        assert!(parse_output("{\"kind\": \"train\"}").is_err());
    }
}
