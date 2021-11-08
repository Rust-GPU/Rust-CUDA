#[cfg(target_os = "windows")]
fn lib_search_path() -> String {
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/nvvm/lib/x64"
        .to_string()
}

#[cfg(target_os = "linux")]
fn lib_search_path() -> String {
    format!("{}/nvvm/lib64", std::env::var("CUDA_ROOT").unwrap())
}

fn libnvvm_build() {
    // on windows, libnvvm should be in CUDA_PATH/nvvm/
    // println!("cargo:rustc-link-lib=dylib=../../../Program Files/NVIDIA GPU
    // Computing Toolkit/CUDA/v11.3/nvvm/bin/nvvm64_40_0");

    let cuda_root = find_cuda_helper::find_cuda_root().expect("Could not figure out cuda installation root");

    #[cfg(windows)]
    let lib_search_path = std::path::Path::new(&cuda_root).join("nvvm").join("lib").join("x64");

    #[cfg(unix)]
    let lib_search_path = std::path::Path::new(&cuda_root).join("nvvm").join("lib64");

    println!("cargo:rustc-link-search={}", lib_search_path.display());
    println!("cargo:rustc-link-lib=dylib=nvvm");
}

fn main() {
    libnvvm_build()
}
