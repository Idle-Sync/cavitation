//! Checks that the output blocks in the guide chapters are what their examples print: the
//! `xor` example in "Backprop by hand", and `load_torch_cnn` and `steering_cnn` in "Image
//! models", so a change to an example that its chapter does not follow fails here.

use std::path::PathBuf;
use std::process::Command;

/// The workspace root, two folders above this crate.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root exists")
}

/// The lines of the chapter's code block after the line `marker`.
fn chapter_output(chapter: &str, marker: &str) -> Vec<String> {
    let after = chapter
        .split(marker)
        .nth(1)
        .unwrap_or_else(|| panic!("the chapter has a '{marker}' line"));
    let block = after
        .split("```text\n")
        .nth(1)
        .expect("a text block follows")
        .split("\n```")
        .next()
        .expect("the block is closed");
    block.lines().map(str::to_string).collect()
}

/// Runs the `xor` example in release mode and compares what it prints with the chapter,
/// line for line.
#[test]
fn xor_chapter_output_matches() {
    let root = workspace_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    // The printed digits can differ between kernel levels (an AVX2 or NEON kernel may round
    // sums differently), so the child runs the reference level on one thread, whatever level
    // the test run itself uses. The chapter shows the reference output.
    let run = Command::new(cargo)
        .current_dir(&root)
        .env("CAVITATION_KERNEL", "reference")
        .env("CAVITATION_THREADS", "1")
        .args([
            "run",
            "--release",
            "--quiet",
            "--example",
            "xor",
            "-p",
            "cavitation",
        ])
        .output()
        .expect("cargo runs");
    assert!(
        run.status.success(),
        "the xor example failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let printed = String::from_utf8(run.stdout).expect("the output is UTF-8");
    // A checkout on Windows may turn the chapter's line endings into CRLF.
    let chapter = std::fs::read_to_string(root.join("book/src/backprop-by-hand.md"))
        .expect("the chapter exists")
        .replace("\r\n", "\n");
    let expected = chapter_output(&chapter, "The whole output:");
    let actual: Vec<&str> = printed.lines().collect();
    for (i, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(want, got, "line {} differs", i + 1);
    }
    assert_eq!(
        expected.len(),
        actual.len(),
        "the chapter has {} lines, the example printed {}",
        expected.len(),
        actual.len()
    );
}

/// Runs `example` at the reference kernel level on one thread, the setting the chapter
/// "Image models" shows, and compares what it prints with the block after `marker` there,
/// line for line.
fn image_chapter_matches(example: &str, marker: &str) {
    let root = workspace_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let run = Command::new(cargo)
        .current_dir(&root)
        .env("CAVITATION_KERNEL", "reference")
        .env("CAVITATION_THREADS", "1")
        .args([
            "run",
            "--release",
            "--quiet",
            "--example",
            example,
            "-p",
            "cavitation",
        ])
        .output()
        .expect("cargo runs");
    assert!(
        run.status.success(),
        "the {example} example failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let printed = String::from_utf8(run.stdout).expect("the output is UTF-8");
    let chapter = std::fs::read_to_string(root.join("book/src/image-models.md"))
        .expect("the chapter exists")
        .replace("\r\n", "\n");
    let expected = chapter_output(&chapter, marker);
    let actual: Vec<&str> = printed.lines().collect();
    assert_eq!(
        expected, actual,
        "{example}'s output differs from the chapter"
    );
}

/// The chapter's `load_torch_cnn` block is what the example prints.
#[test]
fn load_torch_cnn_chapter_output_matches() {
    image_chapter_matches(
        "load_torch_cnn",
        "`load_torch_cnn` prints, at the reference kernel level on one thread:",
    );
}

/// The chapter's `steering_cnn` block is what the example prints.
#[test]
fn steering_cnn_chapter_output_matches() {
    image_chapter_matches(
        "steering_cnn",
        "`steering_cnn` prints, at the reference kernel level on one thread:",
    );
}
