use find_cuda_helper::find_libnvvm_bin_dir;

fn main() {
    println!("cargo:rustc-link-search={}", find_libnvvm_bin_dir());
    println!("cargo:rustc-link-lib=dylib=nvvm");
}
