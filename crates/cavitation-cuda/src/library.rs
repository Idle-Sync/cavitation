//! A dynamic library opened at run time with the operating system's loader, and the
//! functions looked up in it by name.

use std::ffi::{CStr, c_void};
use std::path::Path;
use std::ptr::NonNull;

/// A dynamic library loaded into this process.
///
/// It stays loaded until the process ends: NVIDIA's libraries keep state of their own and
/// load one another by name, so nothing here unloads one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Library {
    /// The loader's handle to the library.
    handle: NonNull<c_void>,
}

impl Library {
    /// Loads the library in the file at `path`, which must be absolute. On Windows the
    /// libraries it imports are looked for in its own folder and the system folder only,
    /// never in the current folder or the folders of `PATH`.
    ///
    /// # Errors
    ///
    /// The loader's message, as text.
    pub(crate) fn open(path: &Path) -> Result<Library, String> {
        if !path.is_absolute() {
            return Err(format!("{} is not an absolute path", path.display()));
        }
        os::open_path(path).map(|handle| Library { handle })
    }

    /// Loads a library by file name from where the operating system keeps its own: the
    /// system folder on Windows, the linker's configured folders on Linux. The folder the
    /// program was started in is never searched.
    ///
    /// # Errors
    ///
    /// The loader's message, as text.
    pub(crate) fn open_system(name: &str) -> Result<Library, String> {
        os::open_system(name).map(|handle| Library { handle })
    }

    /// The function the library exports under `name`, or `None` if it exports none.
    ///
    /// # Safety
    ///
    /// `F` must be an `unsafe extern fn` pointer type with the signature and calling
    /// convention the library gives that function.
    pub(crate) unsafe fn function<F: Copy>(&self, name: &CStr) -> Option<F> {
        assert_eq!(
            size_of::<F>(),
            size_of::<*mut c_void>(),
            "a function pointer type is as wide as a pointer"
        );
        let address = os::symbol(self.handle, name)?;
        // SAFETY: `F` is a function pointer type by this function's contract, which has
        // the size (checked above) and the validity (non-null) of the address looked up.
        Some(unsafe { std::mem::transmute_copy::<NonNull<c_void>, F>(&address) })
    }
}

/// The Windows loader.
#[cfg(windows)]
mod os {
    use std::ffi::{CStr, c_char, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr::NonNull;

    /// `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`: look for the libraries the loaded one imports
    /// in its own folder. Valid only with an absolute path.
    const SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    /// `LOAD_LIBRARY_SEARCH_SYSTEM32`: look in the system folder only.
    const SEARCH_SYSTEM32: u32 = 0x0000_0800;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        /// Loads a library; null on failure, with the reason in the thread's last error.
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        /// The address of an exported function, or null.
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    /// Loads `name` (a path or a file name) with `flags`.
    fn load(name: &std::ffi::OsStr, flags: u32) -> Result<NonNull<c_void>, String> {
        // The loader takes only backslashes as separators in a path it searches beside.
        let mut wide: Vec<u16> = name
            .encode_wide()
            .map(|c| {
                if c == u16::from(b'/') {
                    u16::from(b'\\')
                } else {
                    c
                }
            })
            .collect();
        if wide.contains(&0) {
            return Err(format!("{name:?} holds a NUL character"));
        }
        wide.push(0);
        // SAFETY: `wide` is a NUL-terminated UTF-16 string that lives across the call, the
        // file handle must be null, and the flags are ones the function documents.
        let handle = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) };
        NonNull::new(handle).ok_or_else(|| std::io::Error::last_os_error().to_string())
    }

    /// Loads the library at the absolute `path`; what it imports comes from its own folder
    /// or the system folder, and from nowhere else.
    pub(super) fn open_path(path: &Path) -> Result<NonNull<c_void>, String> {
        load(path.as_os_str(), SEARCH_DLL_LOAD_DIR | SEARCH_SYSTEM32)
    }

    /// Loads `name` from the system folder.
    pub(super) fn open_system(name: &str) -> Result<NonNull<c_void>, String> {
        load(name.as_ref(), SEARCH_SYSTEM32)
    }

    /// The address `handle`'s library exports under `name`.
    pub(super) fn symbol(handle: NonNull<c_void>, name: &CStr) -> Option<NonNull<c_void>> {
        // SAFETY: `handle` came from `LoadLibraryExW` and is never freed, and `name` is a
        // NUL-terminated string that lives across the call.
        NonNull::new(unsafe { GetProcAddress(handle.as_ptr(), name.as_ptr()) })
    }
}

/// The Linux loader.
#[cfg(target_os = "linux")]
mod os {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr::NonNull;

    /// `RTLD_NOW`: resolve every symbol at load time, so a library that cannot run fails
    /// here and not in a later call.
    const NOW: c_int = 0x2;
    /// `RTLD_GLOBAL`: make the library's symbols available to libraries loaded after it,
    /// as NVIDIA's libraries expect of one another.
    const GLOBAL: c_int = 0x100;

    #[link(name = "dl")]
    unsafe extern "C" {
        /// Loads a library; null on failure, with the reason in `dlerror`.
        fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
        /// The address of an exported symbol, or null.
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        /// The last error's message, or null; the text is the loader's own.
        fn dlerror() -> *mut c_char;
    }

    /// A folder of `LD_LIBRARY_PATH` that is empty or relative, which the linker reads
    /// against the current folder, if there is one. An empty value, which the linker
    /// ignores, has none.
    fn relative_loader_folder() -> Option<String> {
        let value = std::env::var_os("LD_LIBRARY_PATH").filter(|v| !v.is_empty())?;
        std::env::split_paths(&value)
            .find(|dir| !dir.is_absolute())
            .map(|dir| format!("{:?}", dir.display().to_string()))
    }

    /// Loads `name`: a path, or a file name the linker looks up in its configured folders.
    /// The linker also looks up the libraries it imports in the folders of
    /// `LD_LIBRARY_PATH`, so nothing is loaded while one of them is empty or relative.
    fn load(name: &[u8]) -> Result<NonNull<c_void>, String> {
        if let Some(dir) = relative_loader_folder() {
            return Err(format!(
                "LD_LIBRARY_PATH holds the folder {dir}, which the linker would read from the \
                 current folder: remove it or make it absolute"
            ));
        }
        let name = CString::new(name).map_err(|_| "the name holds a NUL character")?;
        // SAFETY: `name` is a NUL-terminated string that lives across the call, and the
        // flags are ones `dlopen` documents.
        let handle = unsafe { dlopen(name.as_ptr(), NOW | GLOBAL) };
        NonNull::new(handle).ok_or_else(|| {
            // SAFETY: `dlerror` takes no arguments and returns null or a NUL-terminated
            // string that is valid until the next loader call on this thread.
            let message = unsafe { dlerror() };
            if message.is_null() {
                "the library could not be loaded".to_string()
            } else {
                // SAFETY: non-null, so a NUL-terminated string as described above, copied
                // before any other loader call.
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned()
            }
        })
    }

    /// Loads the library at the absolute `path`.
    pub(super) fn open_path(path: &Path) -> Result<NonNull<c_void>, String> {
        load(path.as_os_str().as_bytes())
    }

    /// Loads `name` from the linker's configured folders: those of `LD_LIBRARY_PATH`, all
    /// absolute (see `load`), and the system's.
    pub(super) fn open_system(name: &str) -> Result<NonNull<c_void>, String> {
        if name.contains('/') {
            return Err(format!("{name:?} is a path, not a file name"));
        }
        load(name.as_bytes())
    }

    /// The address `handle`'s library exports under `name`.
    pub(super) fn symbol(handle: NonNull<c_void>, name: &CStr) -> Option<NonNull<c_void>> {
        // SAFETY: `handle` came from `dlopen` and is never closed, and `name` is a
        // NUL-terminated string that lives across the call.
        NonNull::new(unsafe { dlsym(handle.as_ptr(), name.as_ptr()) })
    }
}

/// Every other platform: NVIDIA publishes no CUDA for it, so nothing loads.
#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    use std::ffi::{CStr, c_void};
    use std::path::Path;
    use std::ptr::NonNull;

    /// The message every load fails with.
    const NO_CUDA: &str = "NVIDIA publishes no CUDA for this platform";

    /// Fails: there is nothing to load.
    pub(super) fn open_path(_path: &Path) -> Result<NonNull<c_void>, String> {
        Err(NO_CUDA.to_string())
    }

    /// Fails: there is nothing to load.
    pub(super) fn open_system(_name: &str) -> Result<NonNull<c_void>, String> {
        Err(NO_CUDA.to_string())
    }

    /// Finds nothing: no library is ever loaded.
    pub(super) fn symbol(_handle: NonNull<c_void>, _name: &CStr) -> Option<NonNull<c_void>> {
        None
    }
}

#[cfg(test)]
mod tests {
    //! Opening a library: a relative path is refused, and a missing file is an error.
    use super::*;

    /// A relative path is refused before the loader sees it, so the folder the program
    /// runs in is never a place a library comes from.
    #[test]
    fn a_relative_path_is_refused() {
        let err = Library::open(Path::new("cudnn64_9.dll")).unwrap_err();
        assert!(err.contains("not an absolute path"), "{err}");
    }

    /// A file that is not there fails with the loader's message, not a panic.
    #[test]
    fn a_missing_file_is_an_error() {
        let path = std::env::temp_dir().join("cavitation-cuda-no-such-library.bin");
        assert!(Library::open(&path).is_err());
        assert!(Library::open_system("cavitation-cuda-no-such-library.bin").is_err());
    }
}
