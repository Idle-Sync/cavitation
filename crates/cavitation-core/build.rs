//! Build script for cavitation-core: links Apple's Accelerate framework when building for
//! Apple Silicon, where `Level::Accelerate` runs matrix products through it. Every other
//! target builds without it.
//!
//! The target is read from Cargo's `CARGO_CFG_TARGET_*` variables, not from `cfg!`, which
//! would describe the machine running the build script instead of the target.

/// Prints the link line for Accelerate on Apple aarch64 targets, and nothing elsewhere.
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if vendor == "apple" && arch == "aarch64" {
        // Accelerate is part of macOS (and of Apple's other systems): no crate, no download.
        println!("cargo::rustc-link-lib=framework=Accelerate");
    }
}
