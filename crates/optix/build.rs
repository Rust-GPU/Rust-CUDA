use std::env;

fn main() {
    let optix_version = env::var("DEP_OPTIX_VERSION")
        .expect("Cannot find transitive metadata 'version' from optix-sys package.")
        .parse::<u32>()
        .expect("Failed to parse OptiX version");

    println!("cargo::rustc-check-cfg=cfg(optix_build_input_instance_array_aabbs)");
    println!("cargo::rustc-check-cfg=cfg(optix_module_compile_options_bound_values)");
    println!("cargo::rustc-check-cfg=cfg(optix_pipeline_compile_options_reserved)");
    println!("cargo::rustc-check-cfg=cfg(optix_program_group_options_reserved)");

    if optix_version < 70200 {
        println!("cargo::rustc-cfg=optix_build_input_instance_array_aabbs");
    }
    if optix_version >= 70200 {
        println!("cargo::rustc-cfg=optix_module_compile_options_bound_values");
    }
    if optix_version >= 70300 {
        println!("cargo::rustc-cfg=optix_pipeline_compile_options_reserved");
        println!("cargo::rustc-cfg=optix_program_group_options_reserved");
    }
}
