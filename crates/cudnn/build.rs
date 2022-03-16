fn main() {
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rerun-if-changed=build.rs");
}
