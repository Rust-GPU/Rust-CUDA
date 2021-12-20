fn main() {
    println!("cargo:include=/usr/local/cuda/include");
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rerun-if-changed=build.rs");
}
