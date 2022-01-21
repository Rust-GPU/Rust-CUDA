use cuda_builder::CudaBuilder;
use find_cuda_helper::find_optix_root;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut optix_include = find_optix_root().expect(
        "Unable to find the OptiX SDK, make sure you installed it and
    that OPTIX_ROOT or OPTIX_ROOT_DIR are set",
    );
    optix_include = optix_include.join("include");

    let args = vec![
        format!("-I{}", optix_include.display()),
        format!("-I{}/../common/gdt", manifest_dir),
    ];

    compile_to_ptx("src/ex02_pipeline.cu", &args);

    let ptx_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("device.ptx");

    CudaBuilder::new("device")
        .copy_to(ptx_path)
        .arch(cuda_builder::NvvmArch::Compute75)
        .optix(true)
        .build()
        .unwrap();
}

fn compile_to_ptx(cu_path: &str, args: &[String]) {
    println!("cargo:rerun-if-changed={}", cu_path);

    let full_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join(cu_path);

    let mut ptx_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(cu_path);
    ptx_path.set_extension("ptx");
    std::fs::create_dir_all(ptx_path.parent().unwrap()).unwrap();

    let output = std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(&full_path)
        .arg("-o")
        .arg(&ptx_path)
        .args(args)
        .output()
        .expect("failed to fun nvcc");

    if !output.status.success() {
        panic!("{}", unsafe { String::from_utf8_unchecked(output.stderr) });
    }
}
