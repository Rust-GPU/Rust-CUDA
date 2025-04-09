use std::env;
use std::path;

pub mod optix_sdk;

// OptiX is a bit exotic in how it provides its functions. It uses a function table
// approach, a function table struct holds function pointers to every optix function. Then
// the Optix driver dll is loaded at runtime and the function table is loaded from that.
// OptiX provides this logic inside optix_stubs.h in the include dir, so we need to compile that
// to a lib and link it in so that we have the initialization and C function logic.
fn main() {
    let sdk = optix_sdk::OptiXSdk::new().expect("Cannot create OptiX SDK instance.");
    let cuda_include_paths = env::var_os("DEP_CUDA_INCLUDES")
        .map(|s| env::split_paths(s.as_os_str()).collect::<Vec<_>>())
        .expect("Cannot find transitive metadata 'cuda_include' from cust_raw package.");

    println!("cargo::rerun-if-changed=build");
    for e in sdk.related_optix_envs() {
        println!("cargo::rerun-if-env-changed={}", e);
    }
    // Emit metadata for the build script.
    println!("cargo::metadata=root={}", sdk.optix_root().display());
    println!("cargo::metadata=version={}", sdk.optix_version());
    println!(
        "cargo::metadata=version_major={}",
        sdk.optix_version_major(),
    );
    println!(
        "cargo::metadata=version_minor={}",
        sdk.optix_version_minor(),
    );
    println!(
        "cargo::metadata=version_micro={}",
        sdk.optix_version_micro(),
    );
    let metadata_optix_include = env::join_paths(sdk.optix_include_paths())
        .map(|s| s.to_string_lossy().to_string())
        .expect("Failed to build metadata for include.");
    println!("cargo::metadata=include_dir={}", metadata_optix_include);

    // Generate optix bindings.
    create_optix_bindings(&sdk, &cuda_include_paths);
    cc::Build::new()
        .file("build/optix_stubs.c")
        .includes(sdk.optix_include_paths())
        .includes(&cuda_include_paths)
        .cpp(false)
        .compile("optix_stubs");
}

fn create_optix_bindings(sdk: &optix_sdk::OptiXSdk, cuda_include_paths: &[path::PathBuf]) {
    let outdir = path::PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR environment variable should be set by cargo."),
    );

    let bindgen_path = path::PathBuf::from(format!("{}/optix_sys.rs", outdir.display()));
    let bindings = bindgen::Builder::default()
        .header("build/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .raw_line("use cust_raw::driver_sys::*;")
        .clang_args(
            sdk.optix_include_paths()
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .clang_args(
            cuda_include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_recursively(false)
        .allowlist_type("Optix.*")
        .allowlist_type("RaygenRecord")
        .allowlist_type("MissRecord")
        .allowlist_type("HitgroupRecord")
        .allowlist_function("optix.*")
        .allowlist_var("OPTIX_VERSION")
        .allowlist_var("OptixSbtRecordHeaderSize")
        .allowlist_var("OptixSbtRecordAlignment")
        .allowlist_var("OptixAccelBufferByteAlignment")
        .allowlist_var("OptixInstanceByteAlignment")
        .allowlist_var("OptixAabbBufferByteAlignment")
        .allowlist_var("OptixGeometryTransformByteAlignment")
        .allowlist_var("OptixTransformByteAlignment")
        .allowlist_var("OptixVersion")
        .allowlist_var("OptixBuildInputSize")
        .allowlist_var("OptixShaderBindingTableSize")
        .newtype_enum("OptixResult")
        .constified_enum_module("OptixCompileOptimizationLevel")
        .constified_enum_module("OptixCompileDebugLevel")
        .constified_enum_module("OptixTraversableGraphFlags")
        .constified_enum_module("OptixExceptionFlags")
        .constified_enum_module("OptixProgramGroupKind")
        .constified_enum_module("OptixDeviceProperty")
        .constified_enum_module("OptixPixelFormat")
        .constified_enum_module("OptixDenoiserModelKind")
        .rustified_enum("GeometryFlags")
        .rustified_enum("OptixGeometryFlags")
        .constified_enum("OptixVertexFormat")
        .constified_enum("OptixIndicesFormat")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .size_t_is_usize(true)
        .layout_tests(true)
        .generate()
        .expect("Unable to generate OptiX bindings");
    bindings
        .write_to_file(bindgen_path.as_path())
        .expect("Cannot write OptiX bindgen output to file.");
}
