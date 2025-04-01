use std::env;

fn main() {
    let cudnn_version = env::var("DEP_CUDNN_VERSION")
        .expect("Cannot find transitive metadata 'version' from cudnn-sys package.")
        .parse::<u32>()
        .expect("Failed to parse cuDNN version");

    println!("cargo::rustc-check-cfg=cfg(cudnn9)");
    if cudnn_version >= 90000 {
        println!("cargo::rustc-cfg=cudnn9");
    }
}
