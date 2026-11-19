//! Why CUDA could not be used, or what one of its calls reported.

use std::fmt;
use std::path::PathBuf;

use crate::find::{PATH_VAR, Platform, loader_var};

/// Why [`Cuda::open`](crate::Cuda::open) found nothing to run on, or what a later call
/// reported.
///
/// Its [`Display`](fmt::Display) form says what is missing and what to install.
///
/// # Examples
///
/// ```
/// use cavitation_cuda::Cuda;
///
/// // On a machine without an NVIDIA card this prints why, and what would change it.
/// if let Err(why) = Cuda::open() {
///     println!("{why}");
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CudaError {
    /// NVIDIA publishes no CUDA for this operating system (macOS, for one).
    Unsupported,
    /// The CUDA driver library could not be loaded: the machine has no NVIDIA display
    /// driver.
    NoDriver {
        /// The driver library's file name.
        library: String,
        /// The loader's message.
        message: String,
    },
    /// The driver loaded and found no CUDA-capable card.
    NoDevice,
    /// The driver and a card are there, and one or more of cuDNN, cuBLAS and NVRTC are not.
    Missing {
        /// One sentence for each library that was not found or did not load, naming the
        /// file looked for.
        problems: Vec<String>,
        /// The folders searched first, in order: those of `CAVITATION_CUDA_PATH`,
        /// `CUDNN_PATH` and `CUDA_PATH`, each with its usual sub-folders.
        searched: Vec<PathBuf>,
        /// The folders searched after them: the ones the operating system's loader is
        /// pointed at (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux).
        loader_folders: Vec<PathBuf>,
    },
    /// A library was found in a version this crate's calls are not written for.
    Version(String),
    /// A call was asked for something it cannot do, such as a convolution where no window
    /// fits, or one cuDNN has no suitable algorithm for.
    Invalid(String),
    /// A library does not export a function this crate calls.
    Symbol {
        /// The library's file name.
        library: String,
        /// The function's name.
        symbol: &'static str,
    },
    /// A call into one of the libraries returned an error status.
    Call {
        /// The function called.
        function: &'static str,
        /// The status it returned.
        code: i32,
        /// The library's name for that status.
        message: String,
    },
    /// NVRTC did not compile a kernel's source.
    Compile {
        /// The name the source was compiled under.
        name: String,
        /// The compiler's log, or its status when it wrote none.
        log: String,
    },
}

impl fmt::Display for CudaError {
    /// Writes what went wrong, and for a missing library what to install.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::Unsupported => {
                write!(f, "NVIDIA publishes no CUDA for this operating system")
            }
            CudaError::NoDriver { library, message } => write!(
                f,
                "no NVIDIA driver: {library} could not be loaded ({message}). CUDA needs an \
                 NVIDIA card and NVIDIA's display driver"
            ),
            CudaError::NoDevice => {
                write!(f, "the NVIDIA driver found no CUDA-capable card")
            }
            CudaError::Missing {
                problems,
                searched,
                loader_folders,
            } => {
                write!(f, "an NVIDIA card is here, but {}", problems.join("; "))?;
                write!(
                    f,
                    ". Install NVIDIA's cuDNN 9 and the CUDA toolkit (12 or 13) it is built \
                     for, or set {PATH_VAR} to the folder that holds these files. Searched: "
                )?;
                let dirs: Vec<String> = searched.iter().map(|d| d.display().to_string()).collect();
                let platform = Platform::here();
                let mut loader = format!(
                    "the {} folder{} of {}",
                    loader_folders.len(),
                    if loader_folders.len() == 1 { "" } else { "s" },
                    loader_var(platform)
                );
                if platform == Platform::Linux {
                    loader.push_str(" and the linker's own");
                }
                match (dirs.is_empty(), loader_folders.is_empty()) {
                    (true, true) => write!(f, "no folder"),
                    (true, false) => write!(f, "{loader}"),
                    (false, true) => write!(f, "{}", dirs.join(", ")),
                    (false, false) => write!(f, "{}, and {loader}", dirs.join(", ")),
                }
            }
            CudaError::Version(message) | CudaError::Invalid(message) => write!(f, "{message}"),
            CudaError::Symbol { library, symbol } => {
                write!(f, "{library} has no function {symbol}")
            }
            CudaError::Call {
                function,
                code,
                message,
            } => write!(f, "{function} failed: {message} (status {code})"),
            CudaError::Compile { name, log } => {
                write!(f, "kernel source {name} did not compile: {log}")
            }
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A missing library's message names the file, what to install, the variable to set
    /// and the folders searched, with the loader's own only counted.
    #[test]
    fn a_missing_library_says_what_to_install() {
        let text = CudaError::Missing {
            problems: vec!["cuDNN 9 (cudnn64_9.dll) was not found".to_string()],
            searched: vec![PathBuf::from("/opt/cuda/lib64")],
            loader_folders: vec![PathBuf::from("/a"), PathBuf::from("/b")],
        }
        .to_string();
        assert!(text.contains("lib64, and the 2 folders of "), "{text}");
        for part in [
            "cudnn64_9.dll",
            "Install NVIDIA's cuDNN 9",
            "CAVITATION_CUDA_PATH",
            "lib64",
        ] {
            assert!(text.contains(part), "{part:?} is not in {text:?}");
        }
        let none = CudaError::Missing {
            problems: vec!["cuBLAS was not found".to_string()],
            searched: Vec::new(),
            loader_folders: Vec::new(),
        };
        assert!(none.to_string().ends_with("Searched: no folder"));
    }

    /// A failed call's message names the function, the status and the library's name for it.
    #[test]
    fn a_failed_call_names_the_function() {
        let text = CudaError::Call {
            function: "cuMemAlloc_v2",
            code: 2,
            message: "CUDA_ERROR_OUT_OF_MEMORY".to_string(),
        }
        .to_string();
        assert_eq!(
            text,
            "cuMemAlloc_v2 failed: CUDA_ERROR_OUT_OF_MEMORY (status 2)"
        );
    }
}
