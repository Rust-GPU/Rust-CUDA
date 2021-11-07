#![cfg_attr(
    target_arch = "nvptx64",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![deny(warnings)]

use cuda_std::*;

/*
#[repr(C)]
struct LaunchParams {
    frame_id: i32,
    color_buffer: *mut u32,
    fb_size: [i32; 2],
}
*/

#[kernel]
pub unsafe fn __closesthit__radiance() {}

#[kernel]
pub unsafe fn __anyhit__radiance() {}

#[kernel]
pub unsafe fn __miss__radiance() {}

#[kernel]
pub unsafe fn __raygen__renderFrame() {
    //crate::println!("Hello from Rust kernel!");
}

// #[kernel]
// pub unsafe fn render(fb: *mut Vec3<f32>, view: &Viewport) {}
