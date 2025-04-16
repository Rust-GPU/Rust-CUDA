use cuda_builder::CudaBuilder;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR exists");
    let manifest_dir = std::path::Path::new(&manifest_dir);
    CudaBuilder::new(manifest_dir.join("../../gpu/add_gpu"))
        .copy_to(manifest_dir.join("../../resources/add.ptx"))
        .build()
        .unwrap();
}
