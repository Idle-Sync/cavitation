//! Checks that the output block in the guide chapter "Backprop by hand" is what the `xor`
//! example prints, so a change to the example that the chapter does not follow fails here.

use std::path::PathBuf;
use std::process::Command;

/// The workspace root, two folders above this crate.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root exists")
}

/// The lines of the chapter's "The whole output:" code block.
fn chapter_output(chapter: &str) -> Vec<String> {
    let after = chapter
        .split("The whole output:")
        .nth(1)
        .expect("the chapter has a 'The whole output:' line");
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
    let expected = chapter_output(&chapter);
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
