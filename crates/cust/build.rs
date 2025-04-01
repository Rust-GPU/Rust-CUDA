use std::env;

fn main() {
    let driver_version = env::var("DEP_CUDA_DRIVER_VERSION")
        .expect("Cannot find transitive metadata 'driver_version' from cust_raw package.")
        .parse::<u32>()
        .expect("Failed to parse CUDA driver version");

    println!("cargo::rustc-check-cfg=cfg(conditional_node)");
    if driver_version >= 12030 {
        println!("cargo::rustc-cfg=conditional_node");
    }
}
