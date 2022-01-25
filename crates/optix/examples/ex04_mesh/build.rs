use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../rust/ex04_mesh_gpu")
        .copy_to("../resources/ex04_mesh.ptx")
        .build()
        .unwrap();
}
