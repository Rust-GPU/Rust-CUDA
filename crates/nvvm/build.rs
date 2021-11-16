use find_cuda_helper::find_cuda_root;

#[cfg(target_os = "windows")]
fn lib_search_path() -> String {
    find_cuda_root()
        .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!")
        .join("nvvm")
        .join("lib")
        .join("x64")
        .to_string_lossy()
        .into_owned()
}

#[cfg(target_os = "linux")]
fn lib_search_path() -> String {
    format!("{}/nvvm/lib64", std::env::var("CUDA_ROOT").unwrap())
}

fn libnvvm_build() {
    println!("cargo:rustc-link-search={}", lib_search_path());
    println!("cargo:rustc-link-lib=dylib=nvvm");
}

fn main() {
    libnvvm_build()
}
