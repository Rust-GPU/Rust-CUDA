use find_cuda_helper::{find_cuda_root, find_optix_root};
use std::env;
use std::path::{Path, PathBuf};

// OptiX is a bit exotic in how it provides its functions. It uses a function table
// approach, a function table struct holds function pointers to every optix function. Then
// the Optix driver dll is loaded at runtime and the function table is loaded from that.
// OptiX provides this logic inside optix_stubs.h in the include dir, so we need to compile that
// to a lib and link it in so that we have the initialization and C function logic.
fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut optix_include = find_optix_root().expect(
        "Unable to find the OptiX SDK, make sure you installed it and
    that OPTIX_ROOT or OPTIX_ROOT_DIR are set",
    );
    optix_include = optix_include.join("include");

    let mut cuda_include = find_cuda_root().expect(
        "Unable to find the CUDA Toolkit, make sure you installed it and
    that CUDA_ROOT, CUDA_PATH or CUDA_TOOLKIT_ROOT_DIR are set",
    );
    cuda_include = cuda_include.join("include");

    bindgen_optix(&optix_include, &cuda_include);

    cc::Build::new()
        .file("./optix_stubs.c")
        .include(optix_include)
        .include(cuda_include)
        .cpp(false)
        .compile("optix_stubs");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=optix_stubs");
}

fn bindgen_optix(optix_include: &Path, cuda_include: &Path) {
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("optix_wrapper.rs");

    let header_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("optix_wrapper.h");

    let this_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("build.rs");

    println!("cargo:rerun-if-changed={}", header_path.display());
    println!("cargo:rerun-if-changed={}", this_path.display());

    let bindings = bindgen::Builder::default()
        .header("src/optix_wrapper.h")
        .clang_arg(format!("-I{}", optix_include.display()))
        .clang_arg(format!("-I{}", cuda_include.display()))
        .allowlist_recursively(false)
        .allowlist_type("Optix.*")
        .allowlist_type("RaygenRecord")
        .allowlist_type("MissRecord")
        .allowlist_type("HitgroupRecord")
        .blocklist_type("OptixBuildInput")
        .allowlist_function("optix.*")
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
        .layout_tests(false)
        .generate_comments(false)
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
        .rust_target(bindgen::RustTarget::nightly())
        .derive_default(true)
        .derive_partialeq(true)
        .formatter(bindgen::Formatter::Rustfmt)
        .generate()
        .expect("Unable to generate optix bindings");

    let dbg_path = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    bindings
        .write_to_file(dbg_path.join("optix_wrapper.rs"))
        .expect("Couldn't write bindings!");

    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
