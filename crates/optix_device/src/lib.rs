#![cfg_attr(
    target_arch = "nvptx64",
    no_std,
    feature(asm_experimental_arch),
    register_attr(nvvm_internal)
)]
use core::arch::asm;

extern crate alloc;

mod hit;
mod intersect;
pub mod misc;
pub mod payload;
mod ray;
pub mod sys;
pub mod trace;
pub mod transform;
pub mod util;

use cuda_std::*;
pub use glam;
use glam::UVec3;

pub use misc::*;

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

/// Functions/items only available in raygen programs (`__raygen__`).
pub mod raygen {
    #[doc(inline)]
    pub use crate::trace::*;
}

/// Functions/items only available in miss programs (`__miss__`).
pub mod intersection {
    #[doc(inline)]
    pub use crate::intersect::{get_attribute, primitive_index, report_intersection};
    #[doc(inline)]
    pub use crate::ray::*;
}

/// Functions/items only available in anyhit programs (`__anyhit__`).
pub mod anyhit {
    #[doc(inline)]
    pub use crate::hit::*;
    #[doc(inline)]
    pub use crate::intersect::{
        get_attribute, ignore_intersection, primitive_index, terminate_ray,
    };
    #[doc(inline)]
    pub use crate::ray::*;
}

/// Functions/items only available in closesthit programs (`__closesthit__`).
pub mod closesthit {
    #[doc(inline)]
    pub use crate::hit::*;
    #[doc(inline)]
    pub use crate::intersect::{get_attribute, primitive_index};
    #[doc(inline)]
    pub use crate::ray::{
        ray_flags, ray_time, ray_tmax, ray_tmin, ray_visibility_mask, ray_world_direction,
        ray_world_origin,
    };
}

/// Functions/items only available in miss programs (`__miss__`).
pub mod miss {
    #[doc(inline)]
    pub use crate::ray::{
        ray_flags, ray_time, ray_tmax, ray_tmin, ray_visibility_mask, ray_world_direction,
        ray_world_origin,
    };
}
