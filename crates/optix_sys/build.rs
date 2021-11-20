use find_cuda_helper::{find_cuda_root, find_optix_root};
use std::env;

// OptiX is a bit exotic in how it provides its functions. It uses a function table
// approach, a function table struct holds function pointers to every optix function. Then
// the Optix driver dll is loaded at runtime and the function table is loaded from that.
// OptiX provides this logic inside optix_stubs.h in the include dir, so we need to compile that
// to a lib and link it in so that we have the initialization and C function logic.
fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut header = find_optix_root().expect(
        "Unable to find the OptiX SDK, make sure you installed it and
    that OPTIX_ROOT or OPTIX_ROOT_DIR are set",
    );
    header = header.join("include");
    let cuda_dir = find_cuda_root()
        .expect(
            "Unable to find the CUDA SDK, make sure you 
    installed it and that CUDA_ROOT is set",
        )
        .join("include");

    cc::Build::new()
        .file("./optix_stubs.c")
        .include(cuda_dir)
        .include(header)
        .cpp(false)
        .compile("optix_stubs");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=optix_stubs");
}
