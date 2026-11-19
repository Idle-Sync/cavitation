//! Where NVIDIA's libraries are looked for and what their files are called. Nothing here
//! loads a library: these are the names and the folders, so they can be tested on a machine
//! that has none of them.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The environment variable naming folders to look in before any other, separated as the
/// platform's `PATH` is.
pub(crate) const PATH_VAR: &str = "CAVITATION_CUDA_PATH";
/// NVIDIA's variable for where cuDNN is installed, when it is set.
const CUDNN_VAR: &str = "CUDNN_PATH";
/// The CUDA toolkit installer's variable for where the toolkit is.
const CUDA_VAR: &str = "CUDA_PATH";

/// The cuDNN major version this crate's calls are written for.
pub(crate) const CUDNN_MAJOR: u32 = 9;
/// The CUDA major versions cuDNN 9 is published for, newest first; cuBLAS and NVRTC carry
/// the same number in their file names.
pub(crate) const CUDA_MAJORS: [u32; 2] = [13, 12];

/// The folders tried under each root a variable names: the root itself and where NVIDIA's
/// installers and archives put libraries below it, each as its path's parts.
const SUBDIRS: [&[&str]; 6] = [
    &[],
    &["bin"],
    &["bin", "x64"],
    &["lib"],
    &["lib64"],
    &["lib", "x64"],
];

/// An operating system, as far as library names and folders differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Platform {
    /// Windows: `name64_N.dll`.
    Windows,
    /// Linux: `libname.so.N`.
    Linux,
    /// Anything else: NVIDIA publishes no CUDA for it.
    Other,
}

impl Platform {
    /// The platform this build runs on.
    pub(crate) fn here() -> Platform {
        if cfg!(windows) {
            Platform::Windows
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else {
            Platform::Other
        }
    }

    /// Whether this is Windows; every other platform is named as Linux names libraries.
    fn windows(self) -> bool {
        self == Platform::Windows
    }
}

/// The CUDA driver library's file name. NVIDIA's display driver installs it.
pub(crate) fn driver_name(platform: Platform) -> &'static str {
    if platform.windows() {
        "nvcuda.dll"
    } else {
        "libcuda.so.1"
    }
}

/// cuDNN's file name: the small library that loads cuDNN's parts.
pub(crate) fn cudnn_name(platform: Platform) -> String {
    if platform.windows() {
        format!("cudnn64_{CUDNN_MAJOR}.dll")
    } else {
        format!("libcudnn.so.{CUDNN_MAJOR}")
    }
}

/// Whether `file` names one of cuDNN's parts (`cudnn_ops`, `cudnn_cnn`, `cudnn_graph` and
/// the rest), which cuDNN loads by name when first called.
pub(crate) fn is_cudnn_part(platform: Platform, file: &str) -> bool {
    if platform.windows() {
        let file = file.to_ascii_lowercase();
        file.starts_with("cudnn_") && file.ends_with(&format!("64_{CUDNN_MAJOR}.dll"))
    } else {
        file.starts_with("libcudnn_") && file.ends_with(&format!(".so.{CUDNN_MAJOR}"))
    }
}

/// cuBLAS's file name for a CUDA major version.
pub(crate) fn cublas_name(platform: Platform, cuda_major: u32) -> String {
    if platform.windows() {
        format!("cublas64_{cuda_major}.dll")
    } else {
        format!("libcublas.so.{cuda_major}")
    }
}

/// NVRTC's file name for a CUDA major version.
pub(crate) fn nvrtc_name(platform: Platform, cuda_major: u32) -> String {
    if platform.windows() {
        format!("nvrtc64_{cuda_major}0_0.dll")
    } else {
        format!("libnvrtc.so.{cuda_major}")
    }
}

/// Whether `file` names NVRTC's built-in functions for a CUDA major version, which NVRTC
/// loads by name when it compiles.
pub(crate) fn is_nvrtc_builtins(platform: Platform, cuda_major: u32, file: &str) -> bool {
    if platform.windows() {
        let file = file.to_ascii_lowercase();
        file.starts_with(&format!("nvrtc-builtins64_{cuda_major}")) && file.ends_with(".dll")
    } else {
        file.starts_with(&format!("libnvrtc-builtins.so.{cuda_major}"))
    }
}

/// The folders to look in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SearchDirs {
    /// Each folder of `CAVITATION_CUDA_PATH`, then of `CUDNN_PATH`, then of `CUDA_PATH`,
    /// and under each the folders in [`SUBDIRS`].
    pub(crate) named: Vec<PathBuf>,
    /// The folders the operating system's loader is pointed at, which are looked in after
    /// the named ones: those of `PATH` on Windows and of `LD_LIBRARY_PATH` elsewhere.
    pub(crate) loader: Vec<PathBuf>,
}

impl SearchDirs {
    /// Every folder, in the order they are looked in.
    pub(crate) fn all(&self) -> Vec<PathBuf> {
        self.named.iter().chain(&self.loader).cloned().collect()
    }
}

/// The variable whose folders the operating system's loader looks in.
pub(crate) fn loader_var(platform: Platform) -> &'static str {
    if platform.windows() {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// The folders to look in, given how to read an environment variable. Only absolute
/// folders are kept, so the folder the program was started in is never one of them, and a
/// folder is listed once.
pub(crate) fn search_dirs(
    platform: Platform,
    var: impl Fn(&str) -> Option<OsString>,
) -> SearchDirs {
    let mut dirs = SearchDirs::default();
    for name in [PATH_VAR, CUDNN_VAR, CUDA_VAR] {
        let Some(value) = var(name) else { continue };
        for root in std::env::split_paths(&value) {
            for sub in SUBDIRS {
                let dir = sub.iter().fold(root.clone(), |dir, part| dir.join(part));
                if dir.is_absolute() && !dirs.named.contains(&dir) {
                    dirs.named.push(dir);
                }
            }
        }
    }
    if let Some(value) = var(loader_var(platform)) {
        for dir in std::env::split_paths(&value) {
            if dir.is_absolute() && !dirs.named.contains(&dir) && !dirs.loader.contains(&dir) {
                dirs.loader.push(dir);
            }
        }
    }
    dirs
}

/// The first of `dirs` that holds a file called `name`, with that file's path.
pub(crate) fn locate(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    dirs.iter()
        .map(|dir| dir.join(name))
        .find(|path| path.is_file())
}

/// The files beside `file` (in its folder) whose names `keep` accepts, sorted by name. A
/// folder that cannot be read gives none.
pub(crate) fn siblings(file: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let Some(dir) = file.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_str().is_some_and(&keep))
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    found.sort();
    found
}

/// cuDNN's parts beside `cudnn`, in an order each can be loaded in: `cudnn_graph`, which
/// the others import, then `cudnn_ops`, which `cudnn_cnn` and `cudnn_adv` import, then the
/// rest by name.
pub(crate) fn cudnn_parts(platform: Platform, cudnn: &Path) -> Vec<PathBuf> {
    let mut parts = siblings(cudnn, |file| is_cudnn_part(platform, file));
    let rank = |path: &PathBuf| {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if name.contains("cudnn_graph") {
            0
        } else if name.contains("cudnn_ops") {
            1
        } else {
            2
        }
    };
    parts.sort_by_key(rank);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new, empty folder under the system's temporary one, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        /// Makes the folder, named after the test and this process.
        fn new(test: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("cavitation-cuda-{test}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("make a temporary folder");
            TempDir(dir)
        }

        /// Makes an empty file in the folder.
        fn touch(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, b"").expect("write a temporary file");
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The file names are the ones NVIDIA ships, per platform and CUDA major version.
    #[test]
    fn names_are_nvidias() {
        let (w, l) = (Platform::Windows, Platform::Linux);
        assert_eq!(driver_name(w), "nvcuda.dll");
        assert_eq!(driver_name(l), "libcuda.so.1");
        assert_eq!(cudnn_name(w), "cudnn64_9.dll");
        assert_eq!(cudnn_name(l), "libcudnn.so.9");
        assert_eq!(cublas_name(w, 13), "cublas64_13.dll");
        assert_eq!(cublas_name(l, 12), "libcublas.so.12");
        assert_eq!(nvrtc_name(w, 13), "nvrtc64_130_0.dll");
        assert_eq!(nvrtc_name(w, 12), "nvrtc64_120_0.dll");
        assert_eq!(nvrtc_name(l, 13), "libnvrtc.so.13");
    }

    /// cuDNN's parts and NVRTC's built-ins are told from their neighbours by name.
    #[test]
    fn parts_are_told_by_name() {
        let (w, l) = (Platform::Windows, Platform::Linux);
        assert!(is_cudnn_part(w, "cudnn_ops64_9.dll"));
        assert!(is_cudnn_part(w, "CUDNN_CNN64_9.DLL"));
        assert!(!is_cudnn_part(w, "cudnn64_9.dll"));
        assert!(!is_cudnn_part(w, "cudnn_ops64_8.dll"));
        assert!(is_cudnn_part(l, "libcudnn_graph.so.9"));
        assert!(!is_cudnn_part(l, "libcudnn.so.9"));
        assert!(!is_cudnn_part(l, "libcudnn_graph.so.9.24.0"));
        assert!(is_nvrtc_builtins(w, 13, "nvrtc-builtins64_130.dll"));
        assert!(!is_nvrtc_builtins(w, 13, "nvrtc-builtins64_129.dll"));
        assert!(!is_nvrtc_builtins(w, 13, "nvrtc64_130_0.dll"));
        assert!(is_nvrtc_builtins(l, 12, "libnvrtc-builtins.so.12.9"));
        assert!(!is_nvrtc_builtins(l, 12, "libnvrtc.so.12"));
    }

    /// The crate's own variable comes first, each root brings its usual folders, the
    /// loader's folders come last, and relative and repeated folders are dropped.
    #[test]
    fn folders_come_in_order() {
        let temp = TempDir::new("order");
        let root = |name: &str| temp.0.join(name);
        let join = |dirs: &[PathBuf]| std::env::join_paths(dirs).expect("join folders");
        let (mine, dnn, cuda, path) = (root("mine"), root("dnn"), root("cuda"), root("path"));
        let var = |name: &str| match name {
            "CAVITATION_CUDA_PATH" => Some(mine.clone().into_os_string()),
            "CUDNN_PATH" => Some(dnn.clone().into_os_string()),
            "CUDA_PATH" => Some(cuda.clone().into_os_string()),
            "PATH" => Some(join(&[
                PathBuf::from("relative"),
                path.clone(),
                mine.clone(),
            ])),
            _ => None,
        };
        let dirs = search_dirs(Platform::Windows, var);
        let n = SUBDIRS.len();
        assert_eq!(dirs.named.len(), 3 * n);
        assert_eq!(dirs.named[0], mine);
        assert_eq!(dirs.named[1], mine.join("bin"));
        assert_eq!(dirs.named[2], mine.join("bin").join("x64"));
        assert_eq!(dirs.named[n], dnn);
        assert_eq!(dirs.named[2 * n], cuda);
        assert_eq!(dirs.loader, std::slice::from_ref(&path));
        assert_eq!(dirs.all().len(), 3 * n + 1);
        assert_eq!(dirs.all()[3 * n], path);
        // Elsewhere the loader's variable is LD_LIBRARY_PATH, so PATH is not read.
        assert!(search_dirs(Platform::Linux, var).loader.is_empty());
        assert_eq!(
            search_dirs(Platform::Linux, |_| None),
            SearchDirs::default()
        );
    }

    /// A file is found in the first folder that holds it, and cuDNN's parts come back with
    /// `cudnn_graph` first and `cudnn_ops` second.
    #[test]
    fn files_are_located() {
        let temp = TempDir::new("locate");
        let empty = temp.0.join("empty");
        std::fs::create_dir_all(&empty).expect("make a folder");
        let cudnn = temp.touch("cudnn64_9.dll");
        for part in ["adv", "cnn", "graph", "ops"] {
            temp.touch(&format!("cudnn_{part}64_9.dll"));
        }
        temp.touch("cublas64_13.dll");
        let dirs = [empty, temp.0.clone()];
        assert_eq!(locate(&dirs, "cudnn64_9.dll"), Some(cudnn.clone()));
        assert_eq!(locate(&dirs, "cublas64_12.dll"), None);
        let names: Vec<String> = cudnn_parts(Platform::Windows, &cudnn)
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "cudnn_graph64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn_cnn64_9.dll"
            ]
        );
    }
}
