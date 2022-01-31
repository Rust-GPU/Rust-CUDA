use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/path_tracer_gpu")
        .copy_to("../../resources/path_tracer.ptx")
        .build()
        .unwrap();

    CudaBuilder::new("../../gpu/path_tracer_gpu")
        .copy_to("../../resources/path_tracer_optix.ptx")
        .build_args(&["--features", "optix"])
        .build()
        .unwrap();
}
