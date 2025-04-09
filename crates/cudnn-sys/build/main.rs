use std::env;
use std::path;

pub mod cudnn_sdk;

fn main() {
    let sdk = cudnn_sdk::CudnnSdk::new().expect("Cannot create cuDNN SDK instance.");
    let cuda_include_paths = env::var_os("DEP_CUDA_INCLUDES")
        .map(|s| env::split_paths(s.as_os_str()).collect::<Vec<_>>())
        .expect("Cannot find transitive metadata 'cuda_include' from cust_raw package.");

    println!("cargo::rerun-if-changed=build");
    // Emit metadata for the build script.
    let (version, version_major, version_minor, version_patch) = (
        sdk.cudnn_version(),
        sdk.cudnn_version_major(),
        sdk.cudnn_version_minor(),
        sdk.cudnn_version_patch(),
    );
    let include_dir = sdk.cudnn_include_path().display().to_string();
    println!("cargo::metadata=version={version}");
    println!("cargo::metadata=version_major={version_major}");
    println!("cargo::metadata=version_minor={version_minor}");
    println!("cargo::metadata=version_patch={version_patch}");
    println!("cargo::metadata=include_dir={include_dir}",);

    // Generate bindings and link to the library.
    create_cudnn_bindings(&sdk, &cuda_include_paths);
    println!("cargo::rustc-link-lib=dylib=cudnn");
}

fn create_cudnn_bindings(sdk: &cudnn_sdk::CudnnSdk, cuda_include_paths: &[path::PathBuf]) {
    println!("cargo::rerun-if-changed=build/wrapper.h");
    let outdir = path::PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR environment variable should be set by cargo."),
    );
    let bindgen_path = path::PathBuf::from(format!("{}/cudnn_sys.rs", outdir.display()));
    let bindings = bindgen::Builder::default()
        .header("build/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_arg(format!("-I{}", sdk.cudnn_include_path().display()))
        .clang_args(
            cuda_include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_function("^cudnn.*")
        .allowlist_type("^cudnn.*")
        .allowlist_var("^CUDNN.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .generate()
        .expect("Unable to generate cuDNN bindings.");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write cuDNN bindgen output to file.");
}
