use find_cuda_helper::{find_cuda_root, find_optix_root};
use std::env;

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

    cc::Build::new()
        .file("./optix_stubs.c")
        .include(optix_include)
        .include(cuda_include)
        .cpp(false)
        .compile("optix_stubs");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=optix_stubs");
}
