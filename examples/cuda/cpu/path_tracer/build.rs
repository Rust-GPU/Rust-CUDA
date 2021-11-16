use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/path_tracer_gpu")
        .copy_to("../../resources/kernels.ptx")
        .build()
        .unwrap();
}
