#![cfg_attr(
    target_arch = "nvptx64",
    no_std,
    feature(register_attr, asm, asm_experimental_arch),
    register_attr(nvvm_internal)
)]

extern crate alloc;

pub mod misc;
pub mod payload;
pub mod sys;
pub mod trace;
pub mod util;

use cuda_std::*;
pub use glam;
use glam::UVec3;

extern "C" {
    pub fn vprintf(format: *const u8, valist: *const core::ffi::c_void) -> i32;
}

#[gpu_only]
#[inline(always)]
pub fn get_launch_index() -> UVec3 {
    let x: u32;
    let y: u32;
    let z: u32;

    unsafe {
        asm!("call ({0}), _optix_get_launch_index_x, ();", out(reg32) x);
        asm!("call ({0}), _optix_get_launch_index_y, ();", out(reg32) y);
        asm!("call ({0}), _optix_get_launch_index_z, ();", out(reg32) z);
    }

    UVec3::new(x, y, z)
}

#[gpu_only]
#[inline(always)]
pub fn get_launch_dimensions() -> UVec3 {
    let x: u32;
    let y: u32;
    let z: u32;

    unsafe {
        asm!("call ({0}), _optix_get_launch_dimension_x, ();", out(reg32) x);
        asm!("call ({0}), _optix_get_launch_dimension_y, ();", out(reg32) y);
        asm!("call ({0}), _optix_get_launch_dimension_z, ();", out(reg32) z);
    }

    UVec3::new(x, y, z)
}

pub mod raygen {
    pub use crate::trace::*;
}
