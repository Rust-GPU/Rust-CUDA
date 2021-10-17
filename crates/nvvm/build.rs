fn libnvvm_build() {
    // on windows, libnvvm should be in CUDA_PATH/nvvm/
    // println!("cargo:rustc-link-lib=dylib=../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/nvvm/bin/nvvm64_40_0");
    println!("cargo:rustc-link-search=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/nvvm/lib/x64");
    println!("cargo:rustc-link-lib=static=nvvm");
}

fn main() {
    libnvvm_build()
}
