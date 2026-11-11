//! Runs the built `cavitation-bench info` and checks that it prints the machine and the
//! forced kernel level.

use std::process::Command;

/// `info` with `CAVITATION_KERNEL=reference` prints the CPU, the core count and that level.
#[test]
fn info_prints_the_forced_level() {
    let out = Command::new(env!("CARGO_BIN_EXE_cavitation-bench"))
        .arg("info")
        .env("CAVITATION_KERNEL", "reference")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("current kernel level: reference"), "{text}");
    for key in [
        "cpu:",
        "logical cores:",
        "detected kernel level:",
        "threads:",
        "rustc:",
    ] {
        assert!(text.contains(key), "missing {key} in {text}");
    }
}
