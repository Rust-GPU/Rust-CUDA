use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    CudaBuilder::new("kernels")
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}
