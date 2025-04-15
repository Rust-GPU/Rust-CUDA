use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("kernels")
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
    CudaBuilder::new("kernels")
        .copy_to(out_path.join("kernels_optix.ptx"))
        .build_args(&["--features", "optix"])
        .build()
        .unwrap();
}
