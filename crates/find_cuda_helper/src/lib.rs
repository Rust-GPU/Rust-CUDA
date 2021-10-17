//! Tiny crate for common logic for finding and including CUDA.

use glob::glob;
use std::{env, path::PathBuf};

pub fn include_cuda() {
    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        for path in find_cuda() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    };

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
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
        path.split(split_char).map(|s| PathBuf::from(s)).collect()
    } else {
        vec![]
    }
}

pub fn find_cuda() -> Vec<PathBuf> {
    let mut candidates = read_env();
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    for e in glob("/usr/local/cuda-*").unwrap() {
        if let Ok(path) = e {
            candidates.push(path)
        }
    }

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
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    valid_paths
}

pub fn find_cuda_windows() -> PathBuf {
    let paths = read_env();
    if !paths.is_empty() {
        return paths[0].clone();
    }

    if let Ok(path) = env::var("CUDA_PATH") {
        // If CUDA_LIBRARY_PATH is not found, then CUDA_PATH will be used when building for
        // Windows to locate the Cuda installation. Cuda installs the full Cuda SDK for 64-bit,
        // but only a limited set of libraries for 32-bit. Namely, it does not include cublas in
        // 32-bit, which cuda-sys requires.

        // 'path' points to the base of the CUDA Installation. The lib directory is a
        // sub-directory.
        let path = PathBuf::from(path);

        // To do this the right way, we check to see which target we're building for.
        let target = env::var("TARGET")
            .expect("cargo did not set the TARGET environment variable as required.");

        // Targets use '-' separators. e.g. x86_64-pc-windows-msvc
        let target_components: Vec<_> = target.as_str().split("-").collect();

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

        // i.e. $CUDA_PATH/lib/x64
        return path.join("lib").join(lib_path);
    }

    // No idea where to look for CUDA
    panic!("Cannot find CUDA library path");
}
