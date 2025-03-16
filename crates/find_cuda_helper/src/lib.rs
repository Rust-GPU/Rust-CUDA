//! Tiny crate for common logic for finding and including CUDA.

use std::{
    env,
    path::{Path, PathBuf},
};

pub fn include_cuda() {
    if env::var("DOCS_RS").is_err() && !cfg!(doc) {
        let paths = find_cuda_lib_dirs();
        if paths.is_empty() {
            panic!("Could not find a cuda installation");
        }
        for path in paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }

        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_ROOT");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
    }
}

// Returns true if the given path is a valid cuda installation
fn is_cuda_root_path<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().join("include").join("cuda.h").is_file()
}

pub fn find_cuda_root() -> Option<PathBuf> {
    // search through the common environment variables first
    for path in ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"]
        .iter()
        .filter_map(|name| std::env::var(*name).ok())
    {
        if is_cuda_root_path(&path) {
            return Some(path.into());
        }
    }

    // If it wasn't specified by env var, try the default installation paths
    #[cfg(not(target_os = "windows"))]
    let default_paths = ["/usr/lib/cuda", "/usr/local/cuda", "/opt/cuda"];
    #[cfg(target_os = "windows")]
    let default_paths = ["C:/CUDA"]; // TODO (AL): what's the actual path here?

    for path in default_paths {
        if is_cuda_root_path(path) {
            return Some(path.into());
        }
    }

    None
}

#[cfg(target_os = "windows")]
pub fn find_cuda_lib_dirs() -> Vec<PathBuf> {
    if let Some(root_path) = find_cuda_root() {
        // To do this the right way, we check to see which target we're building for.
        let target = env::var("TARGET")
            .expect("cargo did not set the TARGET environment variable as required.");

        // Targets use '-' separators. e.g. x86_64-pc-windows-msvc
        let target_components: Vec<_> = target.as_str().split('-').collect();

        // We check that we're building for Windows. This code assumes that the layout in
        // CUDA_PATH matches Windows.
        if target_components[2] != "windows" {
            panic!(
                "The CUDA_PATH variable is only used by cuda-sys on Windows. Your target is {}.",
                target
            );
        }

        // Sanity check that the second component of 'target' is "pc"
        debug_assert_eq!(
            "pc", target_components[1],
            "Expected a Windows target to have the second component be 'pc'. Target: {}",
            target
        );

        // x86_64 should use the libs in the "lib/x64" directory. If we ever support i686 (which
        // does not ship with cublas support), its libraries are in "lib/Win32".
        let lib_path = match target_components[0] {
            "x86_64" => "x64",
            "i686" => {
                // lib path would be "Win32" if we support i686. "cublas" is not present in the
                // 32-bit install.
                panic!("Rust cuda-sys does not currently support 32-bit Windows.");
            }
            _ => {
                panic!("Rust cuda-sys only supports the x86_64 Windows architecture.");
            }
        };

        let lib_dir = root_path.join("lib").join(lib_path);

        return if lib_dir.is_dir() {
            vec![lib_dir]
        } else {
            vec![]
        };
    }

    vec![]
}

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        // The location of the libcuda, libcudart, and libcublas can be hardcoded with the
        // CUDA_LIBRARY_PATH environment variable.
        let split_char = if cfg!(target_os = "windows") {
            ";"
        } else {
            ":"
        };
        path.split(split_char).map(PathBuf::from).collect()
    } else {
        vec![]
    }
}

#[cfg(not(target_os = "windows"))]
pub fn find_cuda_lib_dirs() -> Vec<PathBuf> {
    let mut candidates = read_env();
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    for e in glob::glob("/usr/local/cuda-*").unwrap().flatten() {
        candidates.push(e)
    }
    candidates.push(PathBuf::from("/usr/lib/cuda"));
    candidates.push(detect_cuda_root_via_which_nvcc());

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
            continue;
        }
    }
    valid_paths
}

#[cfg(not(target_os = "windows"))]
fn detect_cuda_root_via_which_nvcc() -> PathBuf {
    use std::process::Command;
    let output = Command::new("which")
        .arg("nvcc")
        .output()
        .expect("Command `which` must be available on *nix like systems.");

    if !output.status.success() {
        panic!("Couldn't find nvcc - `which nvcc` returned non-zero");
    }

    let path: PathBuf = String::from_utf8(output.stdout)
        .expect("Result must be valid UTF-8")
        .trim()
        .to_string()
        .into();

    // The above finds `CUDASDK/bin/nvcc`, so we have to go 2 up for the SDK root.
    path.parent().unwrap().parent().unwrap().to_path_buf()
}

#[cfg(target_os = "windows")]
pub fn find_optix_root() -> Option<PathBuf> {
    // the optix SDK installer sets OPTIX_ROOT_DIR whenever it installs.
    // We also check OPTIX_ROOT first in case someone wants to override it without overriding
    // the SDK-set variable.

    env::var("OPTIX_ROOT")
        .ok()
        .or_else(|| env::var("OPTIX_ROOT_DIR").ok())
        .map(PathBuf::from)
}

#[cfg(target_family = "unix")]
pub fn find_optix_root() -> Option<PathBuf> {
    env::var("OPTIX_ROOT")
        .ok()
        .or_else(|| env::var("OPTIX_ROOT_DIR").ok())
        .map(PathBuf::from)
}

#[cfg(doc)]
pub fn find_libnvvm_bin_dir() -> String {
    String::new()
}

#[cfg(all(target_os = "windows", not(doc)))]
pub fn find_libnvvm_bin_dir() -> String {
    if env::var("DOCS_RS").is_ok() {
        return String::new();
    }
    find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("lib")
        .join("x64")
        .to_string_lossy()
        .into_owned()
}

#[cfg(all(target_os = "linux", not(doc)))]
pub fn find_libnvvm_bin_dir() -> String {
    if env::var("DOCS_RS").is_ok() {
        return String::new();
    }
    find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("lib64")
        .to_string_lossy()
        .into_owned()
}
